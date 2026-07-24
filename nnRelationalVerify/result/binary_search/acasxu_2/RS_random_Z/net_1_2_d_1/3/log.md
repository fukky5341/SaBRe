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
Binary search time: 45.54 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.017083602027241795


# Relational Split (RS_random_Z) starts
Time budget: 1151.87 seconds

## Binary search (step 0) starts
Candidate diff: 0.0994509


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5524182, upper bound: 0.5524182
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5524182, upper bound: 0.5524182
time: 0.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.66 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.66
Output dim: 0, lower bound: -0.5524182, upper bound: 0.5524182
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.66
Output dim: 0, lower bound: -0.5524182, upper bound: 0.5524182
Binary search (step 0): status=Status.VERIFIED, low=0.0994509, high=0.1818182, mid=0.0994509, abs_max=0.6789970397949219
rel_dist={0: [-0.5950600062778149, 0.5950600062778155]}

## Binary search (step 1) starts
Candidate diff: 0.1406345


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5404840, upper bound: 0.5406431
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5406431, upper bound: 0.5404840
time: 0.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.72 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.5404840, upper bound: 0.5406431
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.5406431, upper bound: 0.5404840
Binary search (step 1): status=Status.VERIFIED, low=0.1406345, high=0.1818182, mid=0.1406345, abs_max=0.6789970397949219
rel_dist={0: [-0.5978996472348218, 0.597899647234821]}

## Binary search (step 2) starts
Candidate diff: 0.1612264


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5404840, upper bound: 0.5406431
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5406431, upper bound: 0.5404840
time: 0.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.68 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.5404840, upper bound: 0.5406431
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.5406431, upper bound: 0.5404840
Binary search (step 2): status=Status.VERIFIED, low=0.1612264, high=0.1818182, mid=0.1612264, abs_max=0.6789970397949219
rel_dist={0: [-0.5989763754511993, 0.5989763754511985]}

## Binary search (step 3) starts
Candidate diff: 0.1715223


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5872627, upper bound: 0.5872627
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5872627, upper bound: 0.5957990
time: 0.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.5872627, upper bound: 0.5872627
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.5872627, upper bound: 0.5957990

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5819975, upper bound: 0.5871564
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5957987, upper bound: 0.5824048
time: 0.33 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5872627, upper bound: 0.5957990
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5780913, upper bound: 0.5908405
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.18 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -0.5819975, upper bound: 0.5871564
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -0.5957987, upper bound: 0.5824048
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -0.5872627, upper bound: 0.5957990
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -0.5780913, upper bound: 0.5908405

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5819975, upper bound: 0.5779774
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5427997, upper bound: 0.5871564
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5866687, upper bound: 0.5714988
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5866687, upper bound: 0.5484787
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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5126008, upper bound: 0.5095129
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5120512, upper bound: 0.5095129
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5418400, upper bound: 0.5907920
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5779774, upper bound: 0.5819975
time: 0.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.13 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.13
Output dim: 0, lower bound: -0.5819975, upper bound: 0.5779774
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.13
Output dim: 0, lower bound: -0.5427997, upper bound: 0.5871564
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.13
Output dim: 0, lower bound: -0.5866687, upper bound: 0.5714988
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.13
Output dim: 0, lower bound: -0.5866687, upper bound: 0.5484787
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.13
Output dim: 0, lower bound: -0.5126008, upper bound: 0.5095129
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.13
Output dim: 0, lower bound: -0.5120512, upper bound: 0.5095129
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.13
Output dim: 0, lower bound: -0.5418400, upper bound: 0.5907920
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.13
Output dim: 0, lower bound: -0.5779774, upper bound: 0.5819975

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4967873, upper bound: 0.4751141
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4968696, upper bound: 0.4751141
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4661920, upper bound: 0.4729008
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4661594, upper bound: 0.4729008
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731152, upper bound: 0.5552399
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
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

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5843769, upper bound: 0.5365244
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5866687, upper bound: 0.5484352
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5365244, upper bound: 0.5843769
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5261383, upper bound: 0.5684147
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4702659, upper bound: 0.4700376
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4702659, upper bound: 0.4700376
time: 0.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.4967873, upper bound: 0.4751141
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.4968696, upper bound: 0.4751141
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.4661920, upper bound: 0.4729008
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.4661594, upper bound: 0.4729008
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.5731152, upper bound: 0.5552399
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.5843769, upper bound: 0.5365244
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.5866687, upper bound: 0.5484352
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.5365244, upper bound: 0.5843769
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.5261383, upper bound: 0.5684147
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.4702659, upper bound: 0.4700376
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.4702659, upper bound: 0.4700376

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5650934, upper bound: 0.5224687
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5224687
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
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

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5425344, upper bound: 0.5224687
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5224687
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5425344
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934
time: 0.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.13 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.5650934, upper bound: 0.5224687
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5224687
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.5425344, upper bound: 0.5224687
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5224687
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5425344
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.99 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5729307, upper bound: 0.5508318
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5345663, upper bound: 0.5550195
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 0.97 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5210300, upper bound: 0.5555102
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5273235, upper bound: 0.5534574
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5771027, upper bound: 0.5203719
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5673497, upper bound: 0.5262855
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.98 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5778586, upper bound: 0.5375753
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5285896, upper bound: 0.5394820
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.02 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5149825, upper bound: 0.5254624
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5153839, upper bound: 0.5254624
time: 0.33 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.18 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5729307, upper bound: 0.5508318
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5345663, upper bound: 0.5550195
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5210300, upper bound: 0.5555102
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5273235, upper bound: 0.5534574
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5771027, upper bound: 0.5203719
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5673497, upper bound: 0.5262855
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5778586, upper bound: 0.5375753
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5285896, upper bound: 0.5394820
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5149825, upper bound: 0.5254624
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5153839, upper bound: 0.5254624

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5632065, upper bound: 0.5325406
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5532923, upper bound: 0.5455337
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5671637, upper bound: 0.5262423
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5264616, upper bound: 0.5262574
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5182993, upper bound: 0.5331877
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5485701, upper bound: 0.5294995
time: 0.35 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.19 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -0.5632065, upper bound: 0.5325406
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -0.5532923, upper bound: 0.5455337
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -0.5168306, upper bound: 0.5134462
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.19
Output dim: 0, lower bound: -0.5671637, upper bound: 0.5262423
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -0.5264616, upper bound: 0.5262574
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -0.5182993, upper bound: 0.5331877
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -0.5485701, upper bound: 0.5294995

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5149469, upper bound: 0.5133628
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5149469, upper bound: 0.5133628
time: 0.37 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.26 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.26
Output dim: 0, lower bound: -0.5149469, upper bound: 0.5133628
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.26
Output dim: 0, lower bound: -0.5149469, upper bound: 0.5133628
Binary search (step 3): status=Status.VERIFIED, low=0.1715223, high=0.1818182, mid=0.1715223, abs_max=0.6789970397949219
rel_dist={0: [-0.5994456166514532, 0.5994456166514528]}

## Binary search (step 4) starts
Candidate diff: 0.1766702


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5959558, upper bound: 0.5872627
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5872627, upper bound: 0.5959558
time: 0.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.5959558, upper bound: 0.5872627
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.5872627, upper bound: 0.5959558

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5908405, upper bound: 0.5780913
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5959558, upper bound: 0.5872627
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5681019, upper bound: 0.5868319
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5832535, upper bound: 0.5868319
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.10 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -0.5908405, upper bound: 0.5780913
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -0.5959558, upper bound: 0.5872627
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -0.5681019, upper bound: 0.5868319
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -0.5832535, upper bound: 0.5868319

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4729010, upper bound: 0.4702663
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4729010, upper bound: 0.4702663
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4749358, upper bound: 0.4968808
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4749358, upper bound: 0.4968137
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5639934, upper bound: 0.5442548
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5441623, upper bound: 0.5828222
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

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
time: 0.44 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.20 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.20
Output dim: 0, lower bound: -0.4729010, upper bound: 0.4702663
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.20
Output dim: 0, lower bound: -0.4729010, upper bound: 0.4702663
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.20
Output dim: 0, lower bound: -0.4749358, upper bound: 0.4968808
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.20
Output dim: 0, lower bound: -0.4749358, upper bound: 0.4968137
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.20
Output dim: 0, lower bound: -0.5639934, upper bound: 0.5442548
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 0, lower bound: -0.5441623, upper bound: 0.5828222
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 0, lower bound: -0.5827478, upper bound: 0.5422272
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 0, lower bound: -0.5555948, upper bound: 0.5732267

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5826954
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5439453, upper bound: 0.5719572
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555223, upper bound: 0.5732219
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5445477, upper bound: 0.5653211
time: 0.34 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.78 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5826954
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.5439453, upper bound: 0.5719572
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.5691682, upper bound: 0.5419741
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.5555223, upper bound: 0.5732219
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.78
Output dim: 0, lower bound: -0.5445477, upper bound: 0.5653211

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5395141, upper bound: 0.5780496
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5438800, upper bound: 0.5272497
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

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
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5552588, upper bound: 0.5341016
time: 0.35 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.14 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -0.5395141, upper bound: 0.5780496
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.14
Output dim: 0, lower bound: -0.5438800, upper bound: 0.5272497
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.14
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.14
Output dim: 0, lower bound: -0.5552588, upper bound: 0.5341016

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.99 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5331724, upper bound: 0.5633422
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203876, upper bound: 0.5712291
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5262855, upper bound: 0.5673497
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5771027
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 1.03 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5229830, upper bound: 0.5534167
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5392577, upper bound: 0.5257604
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 1.04 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5220486, upper bound: 0.5177106
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5220486, upper bound: 0.5156195
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

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

Time for candidate selection: 1.00 seconds

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
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.03 seconds

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

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 1.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5218018, upper bound: 0.5248024
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5218018, upper bound: 0.5248024
time: 0.34 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.19 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 0, lower bound: -0.5331724, upper bound: 0.5633422
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -0.5203876, upper bound: 0.5712291
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -0.5262855, upper bound: 0.5673497
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5771027
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 0, lower bound: -0.5229830, upper bound: 0.5534167
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 0, lower bound: -0.5392577, upper bound: 0.5257604
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 0, lower bound: -0.5220486, upper bound: 0.5177106
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 0, lower bound: -0.5220486, upper bound: 0.5156195
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -0.5770248, upper bound: 0.5203789
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -0.5672077, upper bound: 0.5291521
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -0.5677562, upper bound: 0.5203890
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 0, lower bound: -0.5590787, upper bound: 0.5339610
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 0, lower bound: -0.5218018, upper bound: 0.5248024
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 0, lower bound: -0.5218018, upper bound: 0.5248024

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134630, upper bound: 0.5165941
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134637, upper bound: 0.5165941
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5149852
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5149852
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5147432, upper bound: 0.5442714
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5147432, upper bound: 0.5165930
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5503588, upper bound: 0.5147432
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5524152, upper bound: 0.5147510
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for candidate selection: 0.00 seconds

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
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5675479, upper bound: 0.5203276
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5213139, upper bound: 0.5203640
time: 0.39 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.28 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5134630, upper bound: 0.5165941
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5134637, upper bound: 0.5165941
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5149852
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5149852
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5147432, upper bound: 0.5442714
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5147432, upper bound: 0.5165930
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5503588, upper bound: 0.5147432
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5524152, upper bound: 0.5147510
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5670215, upper bound: 0.5291006
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5224842, upper bound: 0.5291295
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5675479, upper bound: 0.5203276
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5213139, upper bound: 0.5203640

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

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

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5308191, upper bound: 0.5147006
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5425415, upper bound: 0.5146982
time: 0.36 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.35 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.35
Output dim: 0, lower bound: -0.5426386, upper bound: 0.5203462
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.35
Output dim: 0, lower bound: -0.5443783, upper bound: 0.5168366
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.35
Output dim: 0, lower bound: -0.5308191, upper bound: 0.5147006
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.35
Output dim: 0, lower bound: -0.5425415, upper bound: 0.5146982
Binary search (step 4): status=Status.VERIFIED, low=0.1766702, high=0.1818182, mid=0.1766702, abs_max=0.6789970397949219
rel_dist={0: [-0.5996024140649461, 0.5996024140649459]}

## Binary search (step 5) starts
Candidate diff: 0.1792442


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5942468, upper bound: 0.5996291
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5996291, upper bound: 0.5942468
time: 0.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.69 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.69
Output dim: 0, lower bound: -0.5942468, upper bound: 0.5996291
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.69
Output dim: 0, lower bound: -0.5996291, upper bound: 0.5942468

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5155116, upper bound: 0.5193790
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5149102, upper bound: 0.5193790
time: 0.31 seconds

## BFS RS instance: RS_RSZ2

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
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5334885, upper bound: 0.5524062
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5334885, upper bound: 0.5524062
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.10 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.10
Output dim: 0, lower bound: -0.5155116, upper bound: 0.5193790
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.10
Output dim: 0, lower bound: -0.5149102, upper bound: 0.5193790
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.10
Output dim: 0, lower bound: -0.5334885, upper bound: 0.5524062
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.10
Output dim: 0, lower bound: -0.5334885, upper bound: 0.5524062
Binary search (step 5): status=Status.VERIFIED, low=0.1792442, high=0.1818182, mid=0.1792442, abs_max=0.6789970397949219
rel_dist={0: [-0.5996291232441683, 0.5996291232441684]}

## Binary search (step 6) starts
Candidate diff: 0.1805312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5709066
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5709066, upper bound: 0.5885052
time: 0.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.66 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5709066
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -0.5709066, upper bound: 0.5885052

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5756191, upper bound: 0.5563882
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5839938, upper bound: 0.5457579
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5709021, upper bound: 0.5847755
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5502593, upper bound: 0.5885052
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.51 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.51
Output dim: 0, lower bound: -0.5756191, upper bound: 0.5563882
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.51
Output dim: 0, lower bound: -0.5839938, upper bound: 0.5457579
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.51
Output dim: 0, lower bound: -0.5709021, upper bound: 0.5847755
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.51
Output dim: 0, lower bound: -0.5502593, upper bound: 0.5885052

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5732267, upper bound: 0.5555948
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5639934, upper bound: 0.5442548
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5561225, upper bound: 0.5455218
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5838626, upper bound: 0.5434509
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5456882, upper bound: 0.5797122
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5563156, upper bound: 0.5756144
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5452494, upper bound: 0.5868760
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5482058, upper bound: 0.5871832
time: 0.33 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.11 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -0.5732267, upper bound: 0.5555948
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.11
Output dim: 0, lower bound: -0.5639934, upper bound: 0.5442548
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.11
Output dim: 0, lower bound: -0.5561225, upper bound: 0.5455218
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -0.5838626, upper bound: 0.5434509
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -0.5456882, upper bound: 0.5797122
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -0.5563156, upper bound: 0.5756144
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -0.5452494, upper bound: 0.5868760
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -0.5482058, upper bound: 0.5871832

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5653211, upper bound: 0.5445477
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5732219, upper bound: 0.5555223
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826954, upper bound: 0.5395691
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5433941, upper bound: 0.5796051
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5454573, upper bound: 0.5292857
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5442129, upper bound: 0.5639256
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555223, upper bound: 0.5732219
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5348471, upper bound: 0.5867507
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5369367, upper bound: 0.5823412
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5348471, upper bound: 0.5870447
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5480742, upper bound: 0.5813899
time: 0.41 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.22
Output dim: 0, lower bound: -0.5653211, upper bound: 0.5445477
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -0.5732219, upper bound: 0.5555223
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -0.5826954, upper bound: 0.5395691
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -0.5433941, upper bound: 0.5796051
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.22
Output dim: 0, lower bound: -0.5454573, upper bound: 0.5292857
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.22
Output dim: 0, lower bound: -0.5442129, upper bound: 0.5639256
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -0.5555223, upper bound: 0.5732219
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -0.5348471, upper bound: 0.5867507
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -0.5369367, upper bound: 0.5823412
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -0.5348471, upper bound: 0.5870447
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -0.5480742, upper bound: 0.5813899

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5341016, upper bound: 0.5552588
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5780496
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5552588, upper bound: 0.5341016
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5027183, upper bound: 0.5027183
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5027183, upper bound: 0.5042602
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5027183, upper bound: 0.5027183
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5027183, upper bound: 0.5027183
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5027183, upper bound: 0.5027183
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5027183, upper bound: 0.5027183
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5096819, upper bound: 0.5027183
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5096819, upper bound: 0.5027183
time: 0.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.5341016, upper bound: 0.5552588
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5780496
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.5552588, upper bound: 0.5341016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.5027183, upper bound: 0.5027183
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.5027183, upper bound: 0.5042602
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.5027183, upper bound: 0.5027183
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.5027183, upper bound: 0.5027183
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.5027183, upper bound: 0.5027183
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.5027183, upper bound: 0.5027183
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.5096819, upper bound: 0.5027183
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.5096819, upper bound: 0.5027183

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for candidate selection: 0.00 seconds

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
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.02 seconds

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

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 1.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5189674, upper bound: 0.5235500
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5527538, upper bound: 0.5229830
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 1.03 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5183419, upper bound: 0.5352427
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5486993, upper bound: 0.5311701
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.05 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5539764, upper bound: 0.5273235
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5580032, upper bound: 0.5219049
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.04 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5735134, upper bound: 0.5398274
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5235405, upper bound: 0.5410208
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 1.04 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5410208, upper bound: 0.5235405
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5340474, upper bound: 0.5735134
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.01 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5331724, upper bound: 0.5633422
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203876, upper bound: 0.5712291
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.04 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5550195, upper bound: 0.5345663
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5508318, upper bound: 0.5729307
time: 0.36 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5634311, upper bound: 0.5365798
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5535169, upper bound: 0.5509407
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5189674, upper bound: 0.5235500
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5527538, upper bound: 0.5229830
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5183419, upper bound: 0.5352427
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5486993, upper bound: 0.5311701
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5539764, upper bound: 0.5273235
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5580032, upper bound: 0.5219049
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5735134, upper bound: 0.5398274
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5235405, upper bound: 0.5410208
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5410208, upper bound: 0.5235405
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5340474, upper bound: 0.5735134
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5331724, upper bound: 0.5633422
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5203876, upper bound: 0.5712291
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5550195, upper bound: 0.5345663
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5508318, upper bound: 0.5729307

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5356729, upper bound: 0.5356884
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5502745, upper bound: 0.5256729
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5256729, upper bound: 0.5502745
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5356884, upper bound: 0.5356729
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134630, upper bound: 0.5165941
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134637, upper bound: 0.5165941
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5183301, upper bound: 0.5247645
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5201653, upper bound: 0.5247645
time: 0.36 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.31 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.31
Output dim: 0, lower bound: -0.5356729, upper bound: 0.5356884
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.31
Output dim: 0, lower bound: -0.5502745, upper bound: 0.5256729
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.31
Output dim: 0, lower bound: -0.5256729, upper bound: 0.5502745
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.31
Output dim: 0, lower bound: -0.5356884, upper bound: 0.5356729
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.31
Output dim: 0, lower bound: -0.5134630, upper bound: 0.5165941
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.31
Output dim: 0, lower bound: -0.5134637, upper bound: 0.5165941
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.31
Output dim: 0, lower bound: -0.5183301, upper bound: 0.5247645
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.31
Output dim: 0, lower bound: -0.5201653, upper bound: 0.5247645
Binary search (step 6): status=Status.VERIFIED, low=0.1805312, high=0.1818182, mid=0.1805312, abs_max=0.6789970397949219
rel_dist={0: [-0.5996332167113175, 0.5996332167113172]}

## Binary search (step 7) starts
Candidate diff: 0.1811747


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5404840, upper bound: 0.5406431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5406431, upper bound: 0.5404840
time: 0.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.75 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.75
Output dim: 0, lower bound: -0.5404840, upper bound: 0.5406431
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.75
Output dim: 0, lower bound: -0.5406431, upper bound: 0.5404840
Binary search (step 7): status=Status.VERIFIED, low=0.1811747, high=0.1818182, mid=0.1811747, abs_max=0.6789970397949219
rel_dist={0: [-0.599635263421194, 0.5996352634211943]}

## Binary search (step 8) starts
Candidate diff: 0.1814964


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5942468, upper bound: 0.5996363
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5996363, upper bound: 0.5942468
time: 0.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -0.5942468, upper bound: 0.5996363
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -0.5996363, upper bound: 0.5942468

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5524062, upper bound: 0.5334885
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5524062, upper bound: 0.5334885
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5481261, upper bound: 0.5942220
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5996067, upper bound: 0.5854029
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.07 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.07
Output dim: 0, lower bound: -0.5524062, upper bound: 0.5334885
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.07
Output dim: 0, lower bound: -0.5524062, upper bound: 0.5334885
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -0.5481261, upper bound: 0.5942220
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -0.5996067, upper bound: 0.5854029

## BFS RS instance: RS_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5427997, upper bound: 0.5871564
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5418400, upper bound: 0.5907920
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5257707, upper bound: 0.5387611
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5257707, upper bound: 0.5387309
time: 0.36 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.20 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 0, lower bound: -0.5427997, upper bound: 0.5871564
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 0, lower bound: -0.5418400, upper bound: 0.5907920
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.20
Output dim: 0, lower bound: -0.5257707, upper bound: 0.5387611
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.20
Output dim: 0, lower bound: -0.5257707, upper bound: 0.5387309

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5369363, upper bound: 0.5585220
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5369367, upper bound: 0.5867507
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4733358, upper bound: 0.4968671
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4731178, upper bound: 0.4967778
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.05 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.05
Output dim: 0, lower bound: -0.5369363, upper bound: 0.5585220
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -0.5369367, upper bound: 0.5867507
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.05
Output dim: 0, lower bound: -0.4733358, upper bound: 0.4968671
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.05
Output dim: 0, lower bound: -0.4731178, upper bound: 0.4967778

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5027183, upper bound: 0.5027183
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5027183, upper bound: 0.5042602
time: 0.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.08 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.5027183, upper bound: 0.5027183
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.5027183, upper bound: 0.5042602
Binary search (step 8): status=Status.VERIFIED, low=0.1814964, high=0.1818182, mid=0.1814964, abs_max=0.6789970397949219
rel_dist={0: [-0.599636286776133, 0.5996362867761327]}

## Binary search (step 9) starts
Candidate diff: 0.1816573


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5709066
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5709066, upper bound: 0.5885052
time: 0.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5709066
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -0.5709066, upper bound: 0.5885052

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5841233, upper bound: 0.5595667
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5883453, upper bound: 0.5707283
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5709021, upper bound: 0.5847755
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5502593, upper bound: 0.5885052
time: 0.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.11 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.11
Output dim: 0, lower bound: -0.5841233, upper bound: 0.5595667
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.11
Output dim: 0, lower bound: -0.5883453, upper bound: 0.5707283
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.11
Output dim: 0, lower bound: -0.5709021, upper bound: 0.5847755
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.11
Output dim: 0, lower bound: -0.5502593, upper bound: 0.5885052

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5813899, upper bound: 0.5585623
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5823412, upper bound: 0.5462201
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5870447, upper bound: 0.5698827
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5867507, upper bound: 0.5466165
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5452494, upper bound: 0.5758741
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5700537, upper bound: 0.5831599
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5125951, upper bound: 0.5066168
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5125951, upper bound: 0.5073438
time: 0.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.04 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.04
Output dim: 0, lower bound: -0.5813899, upper bound: 0.5585623
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.04
Output dim: 0, lower bound: -0.5823412, upper bound: 0.5462201
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.04
Output dim: 0, lower bound: -0.5870447, upper bound: 0.5698827
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.04
Output dim: 0, lower bound: -0.5867507, upper bound: 0.5466165
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.04
Output dim: 0, lower bound: -0.5452494, upper bound: 0.5758741
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.04
Output dim: 0, lower bound: -0.5700537, upper bound: 0.5831599
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.04
Output dim: 0, lower bound: -0.5125951, upper bound: 0.5066168
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.04
Output dim: 0, lower bound: -0.5125951, upper bound: 0.5073438

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5813899, upper bound: 0.5480742
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5369363, upper bound: 0.5585220
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5027191, upper bound: 0.5027191
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5027191, upper bound: 0.5027191
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5552399
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5439457, upper bound: 0.5395691
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5095466, upper bound: 0.5027191
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5095466, upper bound: 0.5027191
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5421801, upper bound: 0.5738801
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5442129, upper bound: 0.5639256
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5052673, upper bound: 0.5027290
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5052673, upper bound: 0.5027290
time: 0.33 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -0.5813899, upper bound: 0.5480742
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.15
Output dim: 0, lower bound: -0.5369363, upper bound: 0.5585220
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.15
Output dim: 0, lower bound: -0.5027191, upper bound: 0.5027191
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.15
Output dim: 0, lower bound: -0.5027191, upper bound: 0.5027191
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.15
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5552399
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.15
Output dim: 0, lower bound: -0.5439457, upper bound: 0.5395691
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.15
Output dim: 0, lower bound: -0.5095466, upper bound: 0.5027191
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.15
Output dim: 0, lower bound: -0.5095466, upper bound: 0.5027191
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -0.5421801, upper bound: 0.5738801
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.15
Output dim: 0, lower bound: -0.5442129, upper bound: 0.5639256
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.15
Output dim: 0, lower bound: -0.5052673, upper bound: 0.5027290
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.15
Output dim: 0, lower bound: -0.5052673, upper bound: 0.5027290

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5439293, upper bound: 0.5443085
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
time: 0.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -0.5439293, upper bound: 0.5443085
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.99 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5221228, upper bound: 0.5186093
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5221228, upper bound: 0.5186093
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 0.97 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5410208, upper bound: 0.5235405
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5398274, upper bound: 0.5735134
time: 0.40 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5221228, upper bound: 0.5186093
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5221228, upper bound: 0.5186093
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5410208, upper bound: 0.5235405
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5398274, upper bound: 0.5735134

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5331610, upper bound: 0.5588753
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203276, upper bound: 0.5675479
time: 0.34 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.14 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 0, lower bound: -0.5331610, upper bound: 0.5588753
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 0, lower bound: -0.5203276, upper bound: 0.5675479

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5138510
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5143041
time: 0.33 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.12 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.12
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5138510
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.12
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5143041
Binary search (step 9): status=Status.VERIFIED, low=0.1816573, high=0.1818182, mid=0.1816573, abs_max=0.6789970397949219
rel_dist={0: [-0.5996367984536023, 0.5996367984536015]}

## Binary search (step 10) starts
Candidate diff: 0.1817378


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5854029, upper bound: 0.5996067
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5996067, upper bound: 0.5854029
time: 0.38 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.5854029, upper bound: 0.5996067
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.5996067, upper bound: 0.5854029

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5819975, upper bound: 0.5871564
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824048, upper bound: 0.5959772
time: 0.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5893335, upper bound: 0.5744534
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5893335, upper bound: 0.5525868
time: 0.32 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.07 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -0.5819975, upper bound: 0.5871564
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -0.5824048, upper bound: 0.5959772
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -0.5893335, upper bound: 0.5744534
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -0.5893335, upper bound: 0.5525868

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4700434, upper bound: 0.4729065
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4700434, upper bound: 0.4729065
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4968106, upper bound: 0.4968730
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4968791, upper bound: 0.4967839
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5866687, upper bound: 0.5714988
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5680253, upper bound: 0.5721867
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5838626, upper bound: 0.5434509
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5561299, upper bound: 0.5463627
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.48 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.48
Output dim: 0, lower bound: -0.4700434, upper bound: 0.4729065
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.48
Output dim: 0, lower bound: -0.4700434, upper bound: 0.4729065
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.48
Output dim: 0, lower bound: -0.4968106, upper bound: 0.4968730
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.48
Output dim: 0, lower bound: -0.4968791, upper bound: 0.4967839
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.48
Output dim: 0, lower bound: -0.5866687, upper bound: 0.5714988
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.48
Output dim: 0, lower bound: -0.5680253, upper bound: 0.5721867
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.48
Output dim: 0, lower bound: -0.5838626, upper bound: 0.5434509
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.48
Output dim: 0, lower bound: -0.5561299, upper bound: 0.5463627

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5439457, upper bound: 0.5552399
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5637699, upper bound: 0.5440061
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5719572
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5225306, upper bound: 0.5395691
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046
time: 0.37 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.56 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.5439457, upper bound: 0.5552399
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.5637699, upper bound: 0.5440061
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5719572
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.5225306, upper bound: 0.5395691
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.5824656, upper bound: 0.5411046

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5438800, upper bound: 0.5272497
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

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
time: 0.36 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.78 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.78
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.78
Output dim: 0, lower bound: -0.5438800, upper bound: 0.5272497
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.98 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5337286, upper bound: 0.5507160
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203878, upper bound: 0.5634436
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.98 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5436455, upper bound: 0.5574925
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5310975, upper bound: 0.5714881
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for candidate selection: 0.00 seconds

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
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 0.98 seconds

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
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.99 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5358548, upper bound: 0.5366994
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5504720, upper bound: 0.5256863
time: 0.37 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.18 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5337286, upper bound: 0.5507160
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5203878, upper bound: 0.5634436
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5436455, upper bound: 0.5574925
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5310975, upper bound: 0.5714881
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5770248, upper bound: 0.5203789
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5672077, upper bound: 0.5291521
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5358548, upper bound: 0.5366994
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.5504720, upper bound: 0.5256863

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5227174, upper bound: 0.5533676
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5185130, upper bound: 0.5202720
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5503588, upper bound: 0.5147432
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5524152, upper bound: 0.5147510
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5670215, upper bound: 0.5291006
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224842, upper bound: 0.5291295
time: 0.36 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.19 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -0.5227174, upper bound: 0.5533676
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -0.5185130, upper bound: 0.5202720
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -0.5503588, upper bound: 0.5147432
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -0.5524152, upper bound: 0.5147510
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.19
Output dim: 0, lower bound: -0.5670215, upper bound: 0.5291006
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -0.5224842, upper bound: 0.5291295

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134549, upper bound: 0.5133983
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134549, upper bound: 0.5133644
time: 0.35 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.31 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -0.5134549, upper bound: 0.5133983
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -0.5134549, upper bound: 0.5133644
Binary search (step 10): status=Status.VERIFIED, low=0.1817378, high=0.1818182, mid=0.1817378, abs_max=0.6789970397949219
rel_dist={0: [-0.5996370542923365, 0.5996370542923366]}

## Binary search (step 11) starts
Candidate diff: 0.1817780


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5709066
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5709066, upper bound: 0.5885052
time: 0.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5709066
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -0.5709066, upper bound: 0.5885052

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
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5871832, upper bound: 0.5700581
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5868760, upper bound: 0.5468601
time: 0.38 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5457579, upper bound: 0.5839938
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5563882, upper bound: 0.5756191
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.16 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -0.5871832, upper bound: 0.5700581
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -0.5868760, upper bound: 0.5468601
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -0.5457579, upper bound: 0.5839938
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -0.5563882, upper bound: 0.5756191

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5732267, upper bound: 0.5555948
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5827478, upper bound: 0.5441623
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5095491, upper bound: 0.5027298
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5095491, upper bound: 0.5027298
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5456882, upper bound: 0.5797122
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5454458, upper bound: 0.5839890
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
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5456882, upper bound: 0.5756144
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5457276, upper bound: 0.5685645
time: 0.33 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.12 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.12
Output dim: 0, lower bound: -0.5732267, upper bound: 0.5555948
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.12
Output dim: 0, lower bound: -0.5827478, upper bound: 0.5441623
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -0.5095491, upper bound: 0.5027298
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -0.5095491, upper bound: 0.5027298
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.12
Output dim: 0, lower bound: -0.5456882, upper bound: 0.5797122
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.12
Output dim: 0, lower bound: -0.5454458, upper bound: 0.5839890
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.12
Output dim: 0, lower bound: -0.5456882, upper bound: 0.5756144
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.12
Output dim: 0, lower bound: -0.5457276, upper bound: 0.5685645

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5653211, upper bound: 0.5445477
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5732219, upper bound: 0.5555223
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5719572, upper bound: 0.5439453
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826954, upper bound: 0.5395691
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5421801, upper bound: 0.5738801
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5440969, upper bound: 0.5781568
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5361240, upper bound: 0.5838579
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5452248, upper bound: 0.5738503
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5560504, upper bound: 0.5754845
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5560577, upper bound: 0.5361471
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5226371, upper bound: 0.5524828
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5445477, upper bound: 0.5653211
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.16
Output dim: 0, lower bound: -0.5653211, upper bound: 0.5445477
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -0.5732219, upper bound: 0.5555223
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -0.5719572, upper bound: 0.5439453
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -0.5826954, upper bound: 0.5395691
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -0.5421801, upper bound: 0.5738801
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -0.5440969, upper bound: 0.5781568
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -0.5361240, upper bound: 0.5838579
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -0.5452248, upper bound: 0.5738503
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -0.5560504, upper bound: 0.5754845
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.16
Output dim: 0, lower bound: -0.5560577, upper bound: 0.5361471
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.16
Output dim: 0, lower bound: -0.5226371, upper bound: 0.5524828
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.16
Output dim: 0, lower bound: -0.5445477, upper bound: 0.5653211

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5341016, upper bound: 0.5552588
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5272497, upper bound: 0.5438800
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
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
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5780496
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5395141, upper bound: 0.5272497
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5439637, upper bound: 0.5637022
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
time: 0.35 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.91 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.5341016, upper bound: 0.5552588
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.5272497, upper bound: 0.5438800
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5780496
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.5395141, upper bound: 0.5272497
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.5439637, upper bound: 0.5637022
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.99 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5248024, upper bound: 0.5218018
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5248024, upper bound: 0.5218018
time: 0.32 seconds

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5714881, upper bound: 0.5310975
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5574925, upper bound: 0.5436455
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 0.99 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5254624, upper bound: 0.5153839
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5254624, upper bound: 0.5150166
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 1.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5183419, upper bound: 0.5352427
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5486993, upper bound: 0.5311701
time: 0.38 seconds

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

Time for candidate selection: 0.00 seconds

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
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 1.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5156558, upper bound: 0.5155758
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5209618, upper bound: 0.5180491
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 1.02 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5189484, upper bound: 0.5248024
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5210413, upper bound: 0.5248024
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 1.03 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5219049, upper bound: 0.5580032
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5235500, upper bound: 0.5539764
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.04 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5150166, upper bound: 0.5254624
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5153839, upper bound: 0.5254624
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.03 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5401287, upper bound: 0.5571939
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5345512, upper bound: 0.5687880
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5373765, upper bound: 0.5542694
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5228855, upper bound: 0.5654198
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.99 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5509407, upper bound: 0.5535169
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5365798, upper bound: 0.5634311
time: 0.36 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5248024, upper bound: 0.5218018
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5248024, upper bound: 0.5218018
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5714881, upper bound: 0.5310975
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5574925, upper bound: 0.5436455
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5254624, upper bound: 0.5153839
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5254624, upper bound: 0.5150166
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5183419, upper bound: 0.5352427
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5486993, upper bound: 0.5311701
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5156558, upper bound: 0.5155758
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5209618, upper bound: 0.5180491
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5189484, upper bound: 0.5248024
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5210413, upper bound: 0.5248024
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5219049, upper bound: 0.5580032
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5235500, upper bound: 0.5539764
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5150166, upper bound: 0.5254624
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5153839, upper bound: 0.5254624
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5401287, upper bound: 0.5571939
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5345512, upper bound: 0.5687880
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5373765, upper bound: 0.5542694
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5228855, upper bound: 0.5654198
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5509407, upper bound: 0.5535169
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.5365798, upper bound: 0.5634311

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5220750, upper bound: 0.5161463
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5220750, upper bound: 0.5157770
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5154704, upper bound: 0.5220009
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5164033, upper bound: 0.5220009
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134690, upper bound: 0.5157945
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134692, upper bound: 0.5157945
time: 0.34 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.27 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -0.5220750, upper bound: 0.5161463
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -0.5220750, upper bound: 0.5157770
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -0.5154704, upper bound: 0.5220009
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -0.5164033, upper bound: 0.5220009
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -0.5134690, upper bound: 0.5157945
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -0.5134692, upper bound: 0.5157945
Binary search (step 11): status=Status.VERIFIED, low=0.1817780, high=0.1818182, mid=0.1817780, abs_max=0.6789970397949219
rel_dist={0: [-0.5996371822117039, 0.5996371822117035]}

## Binary search (step 12) starts
Candidate diff: 0.1817981


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5709066
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5709066, upper bound: 0.5885052
time: 0.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5709066
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.5709066, upper bound: 0.5885052

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5124615, upper bound: 0.5125959
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5124615, upper bound: 0.5125959
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5468601, upper bound: 0.5868760
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5700581, upper bound: 0.5871832
time: 0.33 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.08 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.08
Output dim: 0, lower bound: -0.5124615, upper bound: 0.5125959
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.08
Output dim: 0, lower bound: -0.5124615, upper bound: 0.5125959
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.08
Output dim: 0, lower bound: -0.5468601, upper bound: 0.5868760
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.08
Output dim: 0, lower bound: -0.5700581, upper bound: 0.5871832

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5422272, upper bound: 0.5827478
time: 0.34 seconds

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5096851, upper bound: 0.5027298
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5096851, upper bound: 0.5027298
time: 0.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.09 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.09
Output dim: 0, lower bound: -0.5422272, upper bound: 0.5827478
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.09
Output dim: 0, lower bound: -0.5442548, upper bound: 0.5639934
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.09
Output dim: 0, lower bound: -0.5096851, upper bound: 0.5027298
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.09
Output dim: 0, lower bound: -0.5096851, upper bound: 0.5027298

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5824656
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682
time: 0.33 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.74 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5824656
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.5419741, upper bound: 0.5691682

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
time: 0.37 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.11 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for candidate selection: 0.00 seconds

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
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 0.98 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5156558, upper bound: 0.5155758
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5209618, upper bound: 0.5180491
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 0.99 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5150409, upper bound: 0.5223698
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5157353, upper bound: 0.5223698
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for candidate selection: 0.00 seconds

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
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.01 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5150409, upper bound: 0.5220486
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5177106, upper bound: 0.5220486
time: 0.35 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.19 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 0, lower bound: -0.5156558, upper bound: 0.5155758
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 0, lower bound: -0.5209618, upper bound: 0.5180491
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 0, lower bound: -0.5150409, upper bound: 0.5223698
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 0, lower bound: -0.5157353, upper bound: 0.5223698
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 0, lower bound: -0.5150409, upper bound: 0.5220486
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 0, lower bound: -0.5177106, upper bound: 0.5220486
Binary search (step 12): status=Status.VERIFIED, low=0.1817981, high=0.1818182, mid=0.1817981, abs_max=0.6789970397949219
rel_dist={0: [-0.5996372461476903, 0.5996372461476898]}

## Binary search (step 13) starts
Candidate diff: 0.1818081


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5524182, upper bound: 0.5524182
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5524182, upper bound: 0.5524182
time: 0.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.68 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.5524182, upper bound: 0.5524182
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.5524182, upper bound: 0.5524182
Binary search (step 13): status=Status.VERIFIED, low=0.1818081, high=0.1818182, mid=0.1818081, abs_max=0.6789970397949219
rel_dist={0: [-0.5996372781393807, 0.5996372781393808]}

## Binary search (step 14) starts
Candidate diff: 0.1818132


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5942468, upper bound: 0.5996373
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5996373, upper bound: 0.5942468
time: 0.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.5942468, upper bound: 0.5996373
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.5996373, upper bound: 0.5942468

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5502593
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5502593, upper bound: 0.5847755
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5864481
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5726927
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.15 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.15
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5502593
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.15
Output dim: 0, lower bound: -0.5502593, upper bound: 0.5847755
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.15
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5864481
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.15
Output dim: 0, lower bound: -0.5894953, upper bound: 0.5726927

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5841233, upper bound: 0.5501530
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5883453, upper bound: 0.5406976
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5468357, upper bound: 0.5758741
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5700536, upper bound: 0.5831599
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5403266, upper bound: 0.5862260
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5893335, upper bound: 0.5742962
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5318354, upper bound: 0.5725151
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5893335, upper bound: 0.5525556
time: 0.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.19 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -0.5841233, upper bound: 0.5501530
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -0.5883453, upper bound: 0.5406976
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -0.5468357, upper bound: 0.5758741
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -0.5700536, upper bound: 0.5831599
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -0.5403266, upper bound: 0.5862260
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -0.5893335, upper bound: 0.5742962
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -0.5318354, upper bound: 0.5725151
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -0.5893335, upper bound: 0.5525556

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5463391, upper bound: 0.5454862
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5738503, upper bound: 0.5452248
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5683384, upper bound: 0.5239717
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5838579, upper bound: 0.5361240
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5421801, upper bound: 0.5738801
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5442129, upper bound: 0.5639256
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5698783, upper bound: 0.5826294
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5585220, upper bound: 0.5369363
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5378663, upper bound: 0.5829228
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5365244, upper bound: 0.5843769
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5754845, upper bound: 0.5560504
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5452248, upper bound: 0.5738503
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5239717, upper bound: 0.5454573
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5239717, upper bound: 0.5683384
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5866687, upper bound: 0.5484352
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5759823, upper bound: 0.5491981
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.5463391, upper bound: 0.5454862
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.5738503, upper bound: 0.5452248
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.5683384, upper bound: 0.5239717
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.5838579, upper bound: 0.5361240
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.5421801, upper bound: 0.5738801
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.5442129, upper bound: 0.5639256
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.5698783, upper bound: 0.5826294
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.5585220, upper bound: 0.5369363
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.5378663, upper bound: 0.5829228
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.5365244, upper bound: 0.5843769
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.5754845, upper bound: 0.5560504
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.5452248, upper bound: 0.5738503
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.5239717, upper bound: 0.5454573
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.5239717, upper bound: 0.5683384
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.5866687, upper bound: 0.5484352
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.5759823, upper bound: 0.5491981

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5650934, upper bound: 0.5224687
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5522434, upper bound: 0.5224687
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
time: 0.34 seconds

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
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5395141, upper bound: 0.5780496
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5341016, upper bound: 0.5552588
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5425344
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5637022, upper bound: 0.5439637
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5522434
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5224687
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5443085, upper bound: 0.5439293
time: 0.42 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5650934, upper bound: 0.5224687
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5522434, upper bound: 0.5224687
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5419270, upper bound: 0.5224687
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5395141, upper bound: 0.5780496
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5341016, upper bound: 0.5552588
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5425344
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5731104, upper bound: 0.5551714
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5637022, upper bound: 0.5439637
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5522434
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5780496, upper bound: 0.5395141
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5224687
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5737050, upper bound: 0.5410553
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.5443085, upper bound: 0.5439293

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.01 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5714881, upper bound: 0.5310975
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5574925, upper bound: 0.5436455
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.01 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5634436, upper bound: 0.5203878
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5507160, upper bound: 0.5337286
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.98 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5254624, upper bound: 0.5153839
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5254624, upper bound: 0.5150166
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.02 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5822827, upper bound: 0.5340474
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5285461, upper bound: 0.5340505
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 1.02 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5339610, upper bound: 0.5590787
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203890, upper bound: 0.5677562
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for candidate selection: 0.00 seconds

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
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.01 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5311701, upper bound: 0.5486993
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5352427, upper bound: 0.5183419
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.03 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5550195, upper bound: 0.5345663
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5508318, upper bound: 0.5729307
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 1.01 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5340505, upper bound: 0.5285461
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5340474, upper bound: 0.5822827
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.02 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5150166, upper bound: 0.5254624
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5153839, upper bound: 0.5254624
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.04 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5186014, upper bound: 0.5509550
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5482602, upper bound: 0.5516155
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.05 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5337286, upper bound: 0.5507160
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203878, upper bound: 0.5634436
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.02 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5229830, upper bound: 0.5534167
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5392577, upper bound: 0.5257604
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 1.04 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5183419, upper bound: 0.5352427
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5486993, upper bound: 0.5311701
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 1.06 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5180491, upper bound: 0.5209618
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5155758, upper bound: 0.5156558
time: 0.35 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5714881, upper bound: 0.5310975
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5574925, upper bound: 0.5436455
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5634436, upper bound: 0.5203878
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5507160, upper bound: 0.5337286
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5254624, upper bound: 0.5153839
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5254624, upper bound: 0.5150166
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5822827, upper bound: 0.5340474
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5285461, upper bound: 0.5340505
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5339610, upper bound: 0.5590787
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5203890, upper bound: 0.5677562
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5311701, upper bound: 0.5486993
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5352427, upper bound: 0.5183419
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5550195, upper bound: 0.5345663
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5508318, upper bound: 0.5729307
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5340505, upper bound: 0.5285461
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5340474, upper bound: 0.5822827
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5150166, upper bound: 0.5254624
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5153839, upper bound: 0.5254624
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5186014, upper bound: 0.5509550
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5482602, upper bound: 0.5516155
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5337286, upper bound: 0.5507160
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5203878, upper bound: 0.5634436
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5229830, upper bound: 0.5534167
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5392577, upper bound: 0.5257604
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5183419, upper bound: 0.5352427
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5486993, upper bound: 0.5311701
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5180491, upper bound: 0.5209618
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5155758, upper bound: 0.5156558

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5653871, upper bound: 0.5204377
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5542116, upper bound: 0.5266786
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5538218, upper bound: 0.5272250
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5578253, upper bound: 0.5218353
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203640, upper bound: 0.5213139
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203276, upper bound: 0.5675479
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5462135, upper bound: 0.5480964
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5478817, upper bound: 0.5185589
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5291006, upper bound: 0.5670215
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203224, upper bound: 0.5768506
time: 0.35 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5653871, upper bound: 0.5204377
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5542116, upper bound: 0.5266786
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5538218, upper bound: 0.5272250
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5578253, upper bound: 0.5218353
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5203640, upper bound: 0.5213139
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5203276, upper bound: 0.5675479
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5462135, upper bound: 0.5480964
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5478817, upper bound: 0.5185589
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5291006, upper bound: 0.5670215
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.28
Output dim: 0, lower bound: -0.5203224, upper bound: 0.5768506

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5178927, upper bound: 0.5147959
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5451955, upper bound: 0.5146977
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5146982, upper bound: 0.5425415
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5147006, upper bound: 0.5308191
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133644, upper bound: 0.5134549
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133983, upper bound: 0.5134549
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
time: 0.34 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.25
Output dim: 0, lower bound: -0.5178927, upper bound: 0.5147959
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.25
Output dim: 0, lower bound: -0.5451955, upper bound: 0.5146977
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.25
Output dim: 0, lower bound: -0.5146982, upper bound: 0.5425415
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.25
Output dim: 0, lower bound: -0.5147006, upper bound: 0.5308191
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.25
Output dim: 0, lower bound: -0.5133644, upper bound: 0.5134549
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.25
Output dim: 0, lower bound: -0.5133983, upper bound: 0.5134549
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.25
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.25
Output dim: 0, lower bound: -0.5133628, upper bound: 0.5158407
Binary search (step 14): status=Status.VERIFIED, low=0.1818132, high=0.1818182, mid=0.1818132, abs_max=0.6789970397949219
rel_dist={0: [-0.5996372941589236, 0.5996372941589236]}

## Binary search (step 15) starts
Candidate diff: 0.1818157


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5193791, upper bound: 0.5193791
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5193791, upper bound: 0.5193791
time: 0.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.70 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.70
Output dim: 0, lower bound: -0.5193791, upper bound: 0.5193791
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.70
Output dim: 0, lower bound: -0.5193791, upper bound: 0.5193791
Binary search (step 15): status=Status.VERIFIED, low=0.1818157, high=0.1818182, mid=0.1818157, abs_max=0.6789970397949219
rel_dist={0: [-0.5996373021213002, 0.5996373021213]}

## Binary search (step 16) starts
Candidate diff: 0.1818169


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5942468, upper bound: 0.5996373
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5996373, upper bound: 0.5942468
time: 0.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.5942468, upper bound: 0.5996373
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.5996373, upper bound: 0.5942468

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5502593
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5709021, upper bound: 0.5847755
time: 0.33 seconds

## BFS RS instance: RS_RSZ2

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
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5847755, upper bound: 0.5709021
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5502593, upper bound: 0.5885052
time: 0.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.13 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.13
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5502593
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.13
Output dim: 0, lower bound: -0.5709021, upper bound: 0.5847755
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.13
Output dim: 0, lower bound: -0.5847755, upper bound: 0.5709021
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.13
Output dim: 0, lower bound: -0.5502593, upper bound: 0.5885052

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5685645, upper bound: 0.5457276
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5839890, upper bound: 0.5454458
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5468357, upper bound: 0.5758741
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5700536, upper bound: 0.5831599
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5406827, upper bound: 0.5595304
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5842799, upper bound: 0.5707239
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5406976, upper bound: 0.5883453
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5501530, upper bound: 0.5841233
time: 0.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.17 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.17
Output dim: 0, lower bound: -0.5685645, upper bound: 0.5457276
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.17
Output dim: 0, lower bound: -0.5839890, upper bound: 0.5454458
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.17
Output dim: 0, lower bound: -0.5468357, upper bound: 0.5758741
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.17
Output dim: 0, lower bound: -0.5700536, upper bound: 0.5831599
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.17
Output dim: 0, lower bound: -0.5406827, upper bound: 0.5595304
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.17
Output dim: 0, lower bound: -0.5842799, upper bound: 0.5707239
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.17
Output dim: 0, lower bound: -0.5406976, upper bound: 0.5883453
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.17
Output dim: 0, lower bound: -0.5501530, upper bound: 0.5841233

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5653211, upper bound: 0.5445477
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5524828, upper bound: 0.5226371
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5828174, upper bound: 0.5440248
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5827419, upper bound: 0.5404038
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5465924, upper bound: 0.5756888
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5461953, upper bound: 0.5247751
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5052673, upper bound: 0.5027290
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5052673, upper bound: 0.5027290
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5124583, upper bound: 0.5066066
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5124583, upper bound: 0.5066066
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5361240, upper bound: 0.5838579
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5239717, upper bound: 0.5683384
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5361240, upper bound: 0.5738503
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5239717, upper bound: 0.5463391
time: 0.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.16
Output dim: 0, lower bound: -0.5653211, upper bound: 0.5445477
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.16
Output dim: 0, lower bound: -0.5524828, upper bound: 0.5226371
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.16
Output dim: 0, lower bound: -0.5828174, upper bound: 0.5440248
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.16
Output dim: 0, lower bound: -0.5827419, upper bound: 0.5404038
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.16
Output dim: 0, lower bound: -0.5465924, upper bound: 0.5756888
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.16
Output dim: 0, lower bound: -0.5461953, upper bound: 0.5247751
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.16
Output dim: 0, lower bound: -0.5052673, upper bound: 0.5027290
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.16
Output dim: 0, lower bound: -0.5052673, upper bound: 0.5027290
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.16
Output dim: 0, lower bound: -0.5124583, upper bound: 0.5066066
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.16
Output dim: 0, lower bound: -0.5124583, upper bound: 0.5066066
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.16
Output dim: 0, lower bound: -0.5361240, upper bound: 0.5838579
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.16
Output dim: 0, lower bound: -0.5239717, upper bound: 0.5683384
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.16
Output dim: 0, lower bound: -0.5361240, upper bound: 0.5738503
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.16
Output dim: 0, lower bound: -0.5239717, upper bound: 0.5463391

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5027183, upper bound: 0.5095459
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5027183, upper bound: 0.5095459
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5522434
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214
time: 0.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.84 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -0.5715214, upper bound: 0.5438159
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -0.5826907, upper bound: 0.5308092
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -0.5824606, upper bound: 0.5340679
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -0.5027183, upper bound: 0.5095459
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -0.5027183, upper bound: 0.5095459
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5522434
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -0.5401907, upper bound: 0.5688342
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -0.5438159, upper bound: 0.5715214

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 1.02 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5221228, upper bound: 0.5186093
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5221228, upper bound: 0.5186093
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 0.99 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5189674, upper bound: 0.5235500
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5527538, upper bound: 0.5229830
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 1.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5534574, upper bound: 0.5354946
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5555102, upper bound: 0.5210300
time: 0.35 seconds

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

Time for candidate selection: 0.00 seconds

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
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 1.02 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5539764, upper bound: 0.5273235
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5580032, upper bound: 0.5219049
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5340505, upper bound: 0.5285461
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5340474, upper bound: 0.5822827
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.01 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5262855, upper bound: 0.5673497
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5771027
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.01 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5401287, upper bound: 0.5571939
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5345512, upper bound: 0.5687880
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5

Time for candidate selection: 1.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5186093, upper bound: 0.5221228
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5186093, upper bound: 0.5221228
time: 0.36 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.28 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5221228, upper bound: 0.5186093
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5221228, upper bound: 0.5186093
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5189674, upper bound: 0.5235500
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5527538, upper bound: 0.5229830
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5534574, upper bound: 0.5354946
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5555102, upper bound: 0.5210300
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5539764, upper bound: 0.5273235
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5580032, upper bound: 0.5219049
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5340505, upper bound: 0.5285461
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5340474, upper bound: 0.5822827
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5262855, upper bound: 0.5673497
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5203719, upper bound: 0.5771027
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5401287, upper bound: 0.5571939
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5345512, upper bound: 0.5687880
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5186093, upper bound: 0.5221228
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.5186093, upper bound: 0.5221228

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5149556, upper bound: 0.5223236
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5156396, upper bound: 0.5223236
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5149852
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5149852
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5168306
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5168306
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5154704, upper bound: 0.5220009
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5164033, upper bound: 0.5220009
time: 0.34 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.17 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.17
Output dim: 0, lower bound: -0.5149556, upper bound: 0.5223236
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.17
Output dim: 0, lower bound: -0.5156396, upper bound: 0.5223236
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.17
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5149852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.17
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5149852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.17
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5168306
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.17
Output dim: 0, lower bound: -0.5134462, upper bound: 0.5168306
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.17
Output dim: 0, lower bound: -0.5154704, upper bound: 0.5220009
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.17
Output dim: 0, lower bound: -0.5164033, upper bound: 0.5220009
Binary search (step 16): status=Status.VERIFIED, low=0.1818169, high=0.1818182, mid=0.1818169, abs_max=0.6789970397949219
rel_dist={0: [-0.5996373061498832, 0.5996373061498832]}

## Binary search (step 17) starts
Candidate diff: 0.1818176


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5854029, upper bound: 0.5996067
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5996067, upper bound: 0.5854029
time: 0.38 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.74 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.74
Output dim: 0, lower bound: -0.5854029, upper bound: 0.5996067
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.74
Output dim: 0, lower bound: -0.5996067, upper bound: 0.5854029

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5525868, upper bound: 0.5893335
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5744534, upper bound: 0.5893335
time: 0.34 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5523589, upper bound: 0.5484630
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5523589, upper bound: 0.5484630
time: 0.32 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.14 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -0.5525868, upper bound: 0.5893335
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -0.5744534, upper bound: 0.5893335
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.14
Output dim: 0, lower bound: -0.5523589, upper bound: 0.5484630
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.14
Output dim: 0, lower bound: -0.5523589, upper bound: 0.5484630

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5463627, upper bound: 0.5561299
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5434509, upper bound: 0.5838626
time: 0.34 seconds

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
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5742239, upper bound: 0.5455218
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5561225, upper bound: 0.5754893
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.20
Output dim: 0, lower bound: -0.5463627, upper bound: 0.5561299
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 0, lower bound: -0.5434509, upper bound: 0.5838626
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 0, lower bound: -0.5742239, upper bound: 0.5455218
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 0, lower bound: -0.5561225, upper bound: 0.5754893

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5824656
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5826954
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5225306, upper bound: 0.5439453
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5691682, upper bound: 0.5419741
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5440061, upper bound: 0.5637699
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5552399, upper bound: 0.5731152
time: 0.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.19 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.19
Output dim: 0, lower bound: -0.5411046, upper bound: 0.5824656
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.19
Output dim: 0, lower bound: -0.5395691, upper bound: 0.5826954
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.19
Output dim: 0, lower bound: -0.5225306, upper bound: 0.5439453
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.19
Output dim: 0, lower bound: -0.5691682, upper bound: 0.5419741
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.19
Output dim: 0, lower bound: -0.5440061, upper bound: 0.5637699
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.19
Output dim: 0, lower bound: -0.5552399, upper bound: 0.5731152

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

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
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5395141, upper bound: 0.5780496
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934
time: 0.35 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.53 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.5410553, upper bound: 0.5737050
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.5340679, upper bound: 0.5824606
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.5395141, upper bound: 0.5780496
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.5308092, upper bound: 0.5826907
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.5688342, upper bound: 0.5401907
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5419270
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.5551714, upper bound: 0.5731104
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.5224687, upper bound: 0.5650934

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.01 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5256863, upper bound: 0.5504720
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5366994, upper bound: 0.5358548
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.98 seconds

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
Output dim: 0, lower bound: -0.5203359, upper bound: 0.5770248
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23

Time for candidate selection: 1.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5311701, upper bound: 0.5486993
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5352427, upper bound: 0.5183419
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

Time for candidate selection: 0.00 seconds

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
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 1.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5150166, upper bound: 0.5254624
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5153839, upper bound: 0.5254624
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 1.02 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5634436, upper bound: 0.5203878
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5507160, upper bound: 0.5337286
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.99 seconds

### Candidate
type: RSZ, layer: 3, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5189484, upper bound: 0.5248024
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5218018, upper bound: 0.5248024
time: 0.33 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.5256863, upper bound: 0.5504720
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.5366994, upper bound: 0.5358548
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.5291521, upper bound: 0.5672077
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.5203359, upper bound: 0.5770248
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.5311701, upper bound: 0.5486993
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.5352427, upper bound: 0.5183419
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.5150166, upper bound: 0.5254624
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.5153839, upper bound: 0.5254624
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.5634436, upper bound: 0.5203878
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.5507160, upper bound: 0.5337286
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.5189484, upper bound: 0.5248024
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.5218018, upper bound: 0.5248024

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5291295, upper bound: 0.5224842
time: 0.35 seconds

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

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 5
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5203539, upper bound: 0.5256562
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5203224, upper bound: 0.5768506
time: 0.32 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.18 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 0, lower bound: -0.5291295, upper bound: 0.5224842
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -0.5291006, upper bound: 0.5670215
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 0, lower bound: -0.5203539, upper bound: 0.5256562
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -0.5203224, upper bound: 0.5768506

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

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
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970
1: -0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661
2: -0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567
3: -0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351
4: -0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5146971, upper bound: 0.5522395
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5147006, upper bound: 0.5502034
time: 0.34 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.18 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -0.5133644, upper bound: 0.5134549
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -0.5133983, upper bound: 0.5134549
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -0.5146971, upper bound: 0.5522395
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -0.5147006, upper bound: 0.5502034
Binary search (step 17): status=Status.VERIFIED, low=0.1818176, high=0.1818182, mid=0.1818176, abs_max=0.6789970397949219
rel_dist={0: [-0.5996373101310717, 0.5996373081404776]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.18181755882421413
execution time: 785.10 seconds
