## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_3.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 187.542370087


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746)
1: (-117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561)
2: (-169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212)
3: (-63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962)
4: (-188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602)

## BASE Result
execution time: IAR + LP analysis = 1.78 + 1.75 = 3.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -187.9182065, upper bound: 187.9182065


# Binary Search by BASE starts (time budget: 1196.47 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=232.61239624023438
rel_dist={3: [-187.91820645300623, 187.91820645300623]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=232.61239624023438
rel_dist={3: [-187.90965608592424, 187.9096560859242]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=232.61239624023438
rel_dist={3: [-187.89872093335524, 187.8987209333552]}

## Binary search (step 3) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=232.61239624023438
rel_dist={3: [-187.8886779558913, 187.8886779558913]}

## Binary search (step 4) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=232.61239624023438
rel_dist={3: [-187.88282977162675, 187.88282977162675]}

## Binary search (step 5) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=232.61239624023438
rel_dist={3: [-187.8796990044812, 187.87969900448115]}

## Binary search (step 6) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=232.61239624023438
rel_dist={3: [-187.87806065812612, 187.87806065812612]}

## Binary search (step 7) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=232.61239624023438
rel_dist={3: [-187.87722761678833, 187.87722761678833]}

## Binary search (step 8) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=232.61239624023438
rel_dist={3: [-187.87681109620016, 187.87681109620019]}

## Binary search (step 9) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=232.61239624023438
rel_dist={3: [-187.87660283606715, 187.87660283606715]}

## Binary search (step 10) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=232.61239624023438
rel_dist={3: [-187.87649870699573, 187.87649870699568]}

## Binary search (step 11) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=232.61239624023438
rel_dist={3: [-187.87644665103846, 187.87644665103846]}

## Binary search (step 12) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=232.61239624023438
rel_dist={3: [-187.87642062420437, 187.87642062420434]}

## Binary search (step 13) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=232.61239624023438
rel_dist={3: [-187.87640761299804, 187.87640761299804]}

## Binary search (step 14) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=232.61239624023438
rel_dist={3: [-187.8764011114574, 187.87640150941382]}

## Binary search (step 15) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=232.61239624023438
rel_dist={3: [-187.87639785304535, 187.87639806237894]}

## Binary search (step 16) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=232.61239624023438
rel_dist={3: [-187.87640340339, 187.87639952386695]}

## Binary Search Result
Binary search time: 59.52 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1136.95 seconds

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6535146, upper bound: 187.6590011
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6590011, upper bound: 187.6535146
time: 0.63 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.48 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.48
Output dim: 3, lower bound: -187.6535146, upper bound: 187.6590011
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.48
Output dim: 3, lower bound: -187.6590011, upper bound: 187.6535146

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312887, upper bound: 187.6396430
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312887, upper bound: 187.6316936
time: 0.58 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316936, upper bound: 187.6312887
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396430, upper bound: 187.6312887
time: 0.59 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.02 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 3, lower bound: -187.6312887, upper bound: 187.6396430
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 3, lower bound: -187.6312887, upper bound: 187.6316936
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 3, lower bound: -187.6316936, upper bound: 187.6312887
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 3, lower bound: -187.6396430, upper bound: 187.6312887

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6306595, upper bound: 187.6396426
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312808, upper bound: 187.6396430
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6310797, upper bound: 187.6316936
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312808, upper bound: 187.6269354
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269354, upper bound: 187.6312808
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316936, upper bound: 187.6310797
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396430, upper bound: 187.6312808
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396426, upper bound: 187.6306595
time: 0.58 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.11 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 3, lower bound: -187.6306595, upper bound: 187.6396426
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 3, lower bound: -187.6312808, upper bound: 187.6396430
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 3, lower bound: -187.6310797, upper bound: 187.6316936
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 3, lower bound: -187.6312808, upper bound: 187.6269354
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 3, lower bound: -187.6269354, upper bound: 187.6312808
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 3, lower bound: -187.6316936, upper bound: 187.6310797
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 3, lower bound: -187.6396430, upper bound: 187.6312808
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 3, lower bound: -187.6396426, upper bound: 187.6306595

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280175, upper bound: 187.6392374
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302348, upper bound: 187.6329781
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6286237, upper bound: 187.6392715
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6309060, upper bound: 187.6328802
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282884, upper bound: 187.6308699
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6308965, upper bound: 187.6293496
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6286237, upper bound: 187.6181620
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6309475, upper bound: 187.6198169
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6198169, upper bound: 187.6309475
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6181620, upper bound: 187.6286237
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6293496, upper bound: 187.6308965
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6308699, upper bound: 187.6282884
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6328802, upper bound: 187.6309060
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6392715, upper bound: 187.6286237
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6329781, upper bound: 187.6302348
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6392374, upper bound: 187.6280175
time: 0.63 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 3, lower bound: -187.6280175, upper bound: 187.6392374
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 3, lower bound: -187.6302348, upper bound: 187.6329781
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 3, lower bound: -187.6286237, upper bound: 187.6392715
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 3, lower bound: -187.6309060, upper bound: 187.6328802
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 3, lower bound: -187.6282884, upper bound: 187.6308699
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 3, lower bound: -187.6308965, upper bound: 187.6293496
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 3, lower bound: -187.6286237, upper bound: 187.6181620
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 3, lower bound: -187.6309475, upper bound: 187.6198169
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 3, lower bound: -187.6198169, upper bound: 187.6309475
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 3, lower bound: -187.6181620, upper bound: 187.6286237
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 3, lower bound: -187.6293496, upper bound: 187.6308965
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 3, lower bound: -187.6308699, upper bound: 187.6282884
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 3, lower bound: -187.6328802, upper bound: 187.6309060
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 3, lower bound: -187.6392715, upper bound: 187.6286237
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 3, lower bound: -187.6329781, upper bound: 187.6302348
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 3, lower bound: -187.6392374, upper bound: 187.6280175

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6276140, upper bound: 187.6359014
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272309, upper bound: 187.6383088
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6293701, upper bound: 187.6324970
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6324059
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280153, upper bound: 187.6211107
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282844, upper bound: 187.6383040
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300787, upper bound: 187.6316535
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6292397, upper bound: 187.6323326
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279032, upper bound: 187.6298749
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274766, upper bound: 187.6300658
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300380, upper bound: 187.6290564
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6175484
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280752, upper bound: 187.6169452
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282844, upper bound: 187.6169452
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300938, upper bound: 187.6189678
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289233, upper bound: 187.6169452
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6289233
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6189678, upper bound: 187.6300938
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6282844
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6280752
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6175484, upper bound: 187.6169452
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290564, upper bound: 187.6300380
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300658, upper bound: 187.6274766
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6298749, upper bound: 187.6279032
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6323326, upper bound: 187.6292397
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316535, upper bound: 187.6300787
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6383040, upper bound: 187.6282844
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6211107, upper bound: 187.6280153
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6324059, upper bound: 187.6169452
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6324970, upper bound: 187.6293701
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6383088, upper bound: 187.6272309
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6359014, upper bound: 187.6276140
time: 0.66 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6276140, upper bound: 187.6359014
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6272309, upper bound: 187.6383088
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6293701, upper bound: 187.6324970
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6324059
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6280153, upper bound: 187.6211107
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6282844, upper bound: 187.6383040
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6300787, upper bound: 187.6316535
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6292397, upper bound: 187.6323326
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6279032, upper bound: 187.6298749
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6274766, upper bound: 187.6300658
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6300380, upper bound: 187.6290564
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6175484
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6280752, upper bound: 187.6169452
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6282844, upper bound: 187.6169452
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6300938, upper bound: 187.6189678
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6289233, upper bound: 187.6169452
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6289233
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6189678, upper bound: 187.6300938
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6282844
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6280752
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6175484, upper bound: 187.6169452
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6290564, upper bound: 187.6300380
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6300658, upper bound: 187.6274766
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6298749, upper bound: 187.6279032
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6323326, upper bound: 187.6292397
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6316535, upper bound: 187.6300787
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6383040, upper bound: 187.6282844
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6211107, upper bound: 187.6280153
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6324059, upper bound: 187.6169452
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6324970, upper bound: 187.6293701
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6383088, upper bound: 187.6272309
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.15
Output dim: 3, lower bound: -187.6359014, upper bound: 187.6276140

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274335, upper bound: 187.6359014
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6276140, upper bound: 187.6350569
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272309, upper bound: 187.6383088
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6307631
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6324119
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6293701, upper bound: 187.6324825
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6324057
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6316793
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280153, upper bound: 187.6211107
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6278962, upper bound: 187.6169452
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282844, upper bound: 187.6383040
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6284055, upper bound: 187.6316535
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300787, upper bound: 187.6316265
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289546, upper bound: 187.6323326
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6292292, upper bound: 187.6316206
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277006, upper bound: 187.6298642
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279032, upper bound: 187.6297572
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274766, upper bound: 187.6300658
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6201460
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300380, upper bound: 187.6290339
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6175484
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280752, upper bound: 187.6169452
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280248, upper bound: 187.6169452
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282844, upper bound: 187.6169452
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6284335, upper bound: 187.6169452
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300938, upper bound: 187.6189678
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6287897, upper bound: 187.6169452
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289107, upper bound: 187.6169452
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6289107
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6287897
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6189678, upper bound: 187.6300938
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6284335
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6282844
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6280248
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6280752
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6175484, upper bound: 187.6169452
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290339, upper bound: 187.6300380
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6201460, upper bound: 187.6169452
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300658, upper bound: 187.6274766
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6297572, upper bound: 187.6279032
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6298642, upper bound: 187.6277006
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316206, upper bound: 187.6292292
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6323326, upper bound: 187.6289546
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316265, upper bound: 187.6300787
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316535, upper bound: 187.6284055
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6383040, upper bound: 187.6282844
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6278962
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6211107, upper bound: 187.6280153
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316793, upper bound: 187.6169452
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6324057, upper bound: 187.6169452
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6324825, upper bound: 187.6293701
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6324119, upper bound: 187.6169452
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6307631, upper bound: 187.6169452
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6383088, upper bound: 187.6272309
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6350569, upper bound: 187.6276140
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6359014, upper bound: 187.6274335
time: 0.59 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6274335, upper bound: 187.6359014
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6276140, upper bound: 187.6350569
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6272309, upper bound: 187.6383088
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6307631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6324119
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6293701, upper bound: 187.6324825
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6324057
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6316793
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6280153, upper bound: 187.6211107
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6278962, upper bound: 187.6169452
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6282844, upper bound: 187.6383040
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6284055, upper bound: 187.6316535
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6300787, upper bound: 187.6316265
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6289546, upper bound: 187.6323326
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6292292, upper bound: 187.6316206
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6277006, upper bound: 187.6298642
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6279032, upper bound: 187.6297572
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6274766, upper bound: 187.6300658
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6201460
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6300380, upper bound: 187.6290339
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6175484
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6280752, upper bound: 187.6169452
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6280248, upper bound: 187.6169452
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6282844, upper bound: 187.6169452
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6284335, upper bound: 187.6169452
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6300938, upper bound: 187.6189678
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6287897, upper bound: 187.6169452
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6289107, upper bound: 187.6169452
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6289107
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6287897
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6189678, upper bound: 187.6300938
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6284335
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6282844
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6280248
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6280752
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6175484, upper bound: 187.6169452
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6290339, upper bound: 187.6300380
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6201460, upper bound: 187.6169452
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6300658, upper bound: 187.6274766
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6297572, upper bound: 187.6279032
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6298642, upper bound: 187.6277006
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6316206, upper bound: 187.6292292
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6323326, upper bound: 187.6289546
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6316265, upper bound: 187.6300787
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6316535, upper bound: 187.6284055
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6383040, upper bound: 187.6282844
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6278962
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6211107, upper bound: 187.6280153
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6316793, upper bound: 187.6169452
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6324057, upper bound: 187.6169452
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6324825, upper bound: 187.6293701
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6324119, upper bound: 187.6169452
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6307631, upper bound: 187.6169452
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6383088, upper bound: 187.6272309
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6350569, upper bound: 187.6276140
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -187.6359014, upper bound: 187.6274335

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6359014
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274335, upper bound: 187.6288339
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6350569
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6276140, upper bound: 187.6282481
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6383088
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272309, upper bound: 187.6300629
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6307631
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6324119
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6273223, upper bound: 187.6183649
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168190, upper bound: 187.6290647
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6324057
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6316793
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259671, upper bound: 187.6174273
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168405, upper bound: 187.6203960
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6278962, upper bound: 187.6165894
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260623, upper bound: 187.6181741
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6167864, upper bound: 187.6321630
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6316535
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6284055, upper bound: 187.6169452
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6278500, upper bound: 187.6316265
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300787, upper bound: 187.6169452
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6250123, upper bound: 187.6323326
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289546, upper bound: 187.6169452
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6256475, upper bound: 187.6316206
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6292292, upper bound: 187.6169452
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6298642
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277006, upper bound: 187.6261777
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6297572
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279032, upper bound: 187.6259919
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6256926, upper bound: 187.6186373
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6167864, upper bound: 187.6277597
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6198769
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277836, upper bound: 187.6178818
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168195, upper bound: 187.6266759
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6168405
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6169055
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259949, upper bound: 187.6163102
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168395, upper bound: 187.6163102
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280248, upper bound: 187.6165894
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260623, upper bound: 187.6163102
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168375, upper bound: 187.6163102
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6284335, upper bound: 187.6169452
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6278244, upper bound: 187.6166131
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168268, upper bound: 187.6185118
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6287897, upper bound: 187.6169452
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289107, upper bound: 187.6165894
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6289107
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6287897
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6185118, upper bound: 187.6179519
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6259758
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6284335
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6168375
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6260623
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6280248
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6168405
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6259949
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169055, upper bound: 187.6163102
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168405, upper bound: 187.6163102
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266759, upper bound: 187.6178248
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6178818, upper bound: 187.6277836
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6201460, upper bound: 187.6169452
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6276480, upper bound: 187.6274766
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300658, upper bound: 187.6169452
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6261006, upper bound: 187.6279032
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6297572, upper bound: 187.6169452
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6265049, upper bound: 187.6277006
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6298642, upper bound: 187.6169452
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6292292
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316206, upper bound: 187.6256475
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6289546
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6323326, upper bound: 187.6250123
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6300787
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316265, upper bound: 187.6278500
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.99 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=232.61239624023438
rel_dist={3: [-187.91820645300623, 187.91820645300623]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6528264, upper bound: 187.6571362
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6571362, upper bound: 187.6528264
time: 0.62 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.35 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.35
Output dim: 3, lower bound: -187.6528264, upper bound: 187.6571362
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.35
Output dim: 3, lower bound: -187.6571362, upper bound: 187.6528264

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6307839, upper bound: 187.6375296
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6308252, upper bound: 187.6307779
time: 0.63 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6307779, upper bound: 187.6308252
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6375296, upper bound: 187.6307839
time: 0.68 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.06 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 3, lower bound: -187.6307839, upper bound: 187.6375296
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 3, lower bound: -187.6308252, upper bound: 187.6307779
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 3, lower bound: -187.6307779, upper bound: 187.6308252
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 3, lower bound: -187.6375296, upper bound: 187.6307839

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6303262, upper bound: 187.6375296
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6307839, upper bound: 187.6374853
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6308093, upper bound: 187.6307779
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6308246, upper bound: 187.6261189
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6261189, upper bound: 187.6308246
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6307779, upper bound: 187.6308093
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6374853, upper bound: 187.6307839
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6375296, upper bound: 187.6303262
time: 0.63 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.07 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 3, lower bound: -187.6303262, upper bound: 187.6375296
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 3, lower bound: -187.6307839, upper bound: 187.6374853
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 3, lower bound: -187.6308093, upper bound: 187.6307779
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 3, lower bound: -187.6308246, upper bound: 187.6261189
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 3, lower bound: -187.6261189, upper bound: 187.6308246
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 3, lower bound: -187.6307779, upper bound: 187.6308093
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 3, lower bound: -187.6374853, upper bound: 187.6307839
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 3, lower bound: -187.6375296, upper bound: 187.6303262

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6275700, upper bound: 187.6366910
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300549, upper bound: 187.6310212
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279000, upper bound: 187.6365805
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6303796, upper bound: 187.6309179
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277316, upper bound: 187.6301869
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6306024, upper bound: 187.6283205
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279000, upper bound: 187.6181309
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6306360, upper bound: 187.6197498
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6197498, upper bound: 187.6306360
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6181309, upper bound: 187.6279000
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6283205, upper bound: 187.6306024
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301869, upper bound: 187.6277316
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6309179, upper bound: 187.6303796
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6365805, upper bound: 187.6279000
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6310212, upper bound: 187.6300549
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6366910, upper bound: 187.6275700
time: 0.75 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6275700, upper bound: 187.6366910
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6300549, upper bound: 187.6310212
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6279000, upper bound: 187.6365805
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6303796, upper bound: 187.6309179
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6277316, upper bound: 187.6301869
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6306024, upper bound: 187.6283205
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6279000, upper bound: 187.6181309
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6306360, upper bound: 187.6197498
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6197498, upper bound: 187.6306360
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6181309, upper bound: 187.6279000
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6283205, upper bound: 187.6306024
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6301869, upper bound: 187.6277316
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6309179, upper bound: 187.6303796
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6365805, upper bound: 187.6279000
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6310212, upper bound: 187.6300549
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6366910, upper bound: 187.6275700

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6270128, upper bound: 187.6332789
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6267290, upper bound: 187.6358477
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291522, upper bound: 187.6306148
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6303190
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271963, upper bound: 187.6211107
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272986, upper bound: 187.6357161
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6295640, upper bound: 187.6298030
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6285196, upper bound: 187.6302788
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271707, upper bound: 187.6290468
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268537, upper bound: 187.6294001
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6296993, upper bound: 187.6279118
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6175484
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272284, upper bound: 187.6169452
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272986, upper bound: 187.6169452
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6297596, upper bound: 187.6189678
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277927, upper bound: 187.6169452
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6277927
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6189678, upper bound: 187.6297596
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6272986
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6272284
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6175484, upper bound: 187.6169452
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279118, upper bound: 187.6296993
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6294001, upper bound: 187.6268537
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290468, upper bound: 187.6271707
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302788, upper bound: 187.6285196
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6298030, upper bound: 187.6295640
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6357161, upper bound: 187.6272986
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6211107, upper bound: 187.6271963
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6303190, upper bound: 187.6169452
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6306148, upper bound: 187.6291522
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6358477, upper bound: 187.6267290
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6332789, upper bound: 187.6270128
time: 0.57 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6270128, upper bound: 187.6332789
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6267290, upper bound: 187.6358477
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6291522, upper bound: 187.6306148
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6303190
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6271963, upper bound: 187.6211107
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6272986, upper bound: 187.6357161
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6295640, upper bound: 187.6298030
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6285196, upper bound: 187.6302788
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6271707, upper bound: 187.6290468
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6268537, upper bound: 187.6294001
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6296993, upper bound: 187.6279118
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6175484
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6272284, upper bound: 187.6169452
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6272986, upper bound: 187.6169452
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6297596, upper bound: 187.6189678
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6277927, upper bound: 187.6169452
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6277927
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6189678, upper bound: 187.6297596
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6272986
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6272284
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6175484, upper bound: 187.6169452
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6279118, upper bound: 187.6296993
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6294001, upper bound: 187.6268537
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6290468, upper bound: 187.6271707
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6302788, upper bound: 187.6285196
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6298030, upper bound: 187.6295640
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6357161, upper bound: 187.6272986
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6211107, upper bound: 187.6271963
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6303190, upper bound: 187.6169452
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6306148, upper bound: 187.6291522
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6358477, upper bound: 187.6267290
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.12
Output dim: 3, lower bound: -187.6332789, upper bound: 187.6270128

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268777, upper bound: 187.6332789
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6270114, upper bound: 187.6326328
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6267276, upper bound: 187.6358477
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6292153
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6303513
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291522, upper bound: 187.6306078
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6303190
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6297453
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271949, upper bound: 187.6211107
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271700, upper bound: 187.6169452
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272971, upper bound: 187.6357161
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274336, upper bound: 187.6298030
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6295640, upper bound: 187.6296988
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6281606, upper bound: 187.6302788
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6285131, upper bound: 187.6296686
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6270188, upper bound: 187.6290411
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271693, upper bound: 187.6288528
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268523, upper bound: 187.6294001
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6201460
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6296993, upper bound: 187.6279028
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6175484
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272270, upper bound: 187.6169452
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272101, upper bound: 187.6169452
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272971, upper bound: 187.6169452
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274513, upper bound: 187.6169452
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6297596, upper bound: 187.6189678
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6276790, upper bound: 187.6169452
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277913, upper bound: 187.6169452
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6277913
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6276790
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6189678, upper bound: 187.6297596
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6274513
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6272971
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6272101
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6272270
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6175484, upper bound: 187.6169452
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279028, upper bound: 187.6296993
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6201460, upper bound: 187.6169452
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6294001, upper bound: 187.6268523
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6288528, upper bound: 187.6271693
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290411, upper bound: 187.6270188
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6296686, upper bound: 187.6285131
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302788, upper bound: 187.6281606
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6296988, upper bound: 187.6295640
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6298030, upper bound: 187.6274336
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6357161, upper bound: 187.6272971
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6271700
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6211107, upper bound: 187.6271949
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6297453, upper bound: 187.6169452
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6303190, upper bound: 187.6169452
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6306078, upper bound: 187.6291522
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6303513, upper bound: 187.6169452
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6292153, upper bound: 187.6169452
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6358477, upper bound: 187.6267276
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6326328, upper bound: 187.6270114
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6332789, upper bound: 187.6268777
time: 0.58 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6268777, upper bound: 187.6332789
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6270114, upper bound: 187.6326328
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6267276, upper bound: 187.6358477
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6292153
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6303513
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6291522, upper bound: 187.6306078
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6303190
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6297453
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6271949, upper bound: 187.6211107
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6271700, upper bound: 187.6169452
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6272971, upper bound: 187.6357161
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6274336, upper bound: 187.6298030
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6295640, upper bound: 187.6296988
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6281606, upper bound: 187.6302788
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6285131, upper bound: 187.6296686
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6270188, upper bound: 187.6290411
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6271693, upper bound: 187.6288528
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6268523, upper bound: 187.6294001
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6201460
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6296993, upper bound: 187.6279028
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6175484
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6272270, upper bound: 187.6169452
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6272101, upper bound: 187.6169452
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6272971, upper bound: 187.6169452
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6274513, upper bound: 187.6169452
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6297596, upper bound: 187.6189678
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6276790, upper bound: 187.6169452
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6277913, upper bound: 187.6169452
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6277913
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6276790
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6189678, upper bound: 187.6297596
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6274513
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6272971
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6272101
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6272270
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6175484, upper bound: 187.6169452
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6279028, upper bound: 187.6296993
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6201460, upper bound: 187.6169452
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6294001, upper bound: 187.6268523
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6288528, upper bound: 187.6271693
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6290411, upper bound: 187.6270188
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6296686, upper bound: 187.6285131
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6302788, upper bound: 187.6281606
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6296988, upper bound: 187.6295640
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6298030, upper bound: 187.6274336
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6357161, upper bound: 187.6272971
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6271700
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6211107, upper bound: 187.6271949
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6297453, upper bound: 187.6169452
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6303190, upper bound: 187.6169452
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6306078, upper bound: 187.6291522
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6303513, upper bound: 187.6169452
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6292153, upper bound: 187.6169452
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6358477, upper bound: 187.6267276
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6326328, upper bound: 187.6270114
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 3, lower bound: -187.6332789, upper bound: 187.6268777

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6332789
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268777, upper bound: 187.6272392
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6326328
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6270114, upper bound: 187.6268330
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6358477
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6267276, upper bound: 187.6278937
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6292153
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6303513
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272750, upper bound: 187.6183649
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6178204, upper bound: 187.6290647
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6303190
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6297453
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6169452
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259671, upper bound: 187.6174273
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168405, upper bound: 187.6203960
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271700, upper bound: 187.6165894
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260623, upper bound: 187.6181741
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6171180, upper bound: 187.6319442
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6298030
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274336, upper bound: 187.6169452
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271599, upper bound: 187.6296988
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6295640, upper bound: 187.6169452
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6245421, upper bound: 187.6302788
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6281606, upper bound: 187.6169452
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6251096, upper bound: 187.6296686
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6285131, upper bound: 187.6169452
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6290411
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6270188, upper bound: 187.6254669
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6288528
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271693, upper bound: 187.6252971
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6294001
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268523, upper bound: 187.6258914
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6198769
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277106, upper bound: 187.6178620
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6178248, upper bound: 187.6266363
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6168405
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6169055
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259949, upper bound: 187.6163102
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168405, upper bound: 187.6163102
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272101, upper bound: 187.6165894
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260623, upper bound: 187.6163102
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168375, upper bound: 187.6163102
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274513, upper bound: 187.6165894
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260687, upper bound: 187.6187512
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6297596, upper bound: 187.6165894
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6276790, upper bound: 187.6165894
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277913, upper bound: 187.6165894
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6277913
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6276790
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6297596
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6187512, upper bound: 187.6260687
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6274513
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6168375
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6260623
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6272101
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6168405
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6259949
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169055, upper bound: 187.6163102
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168410, upper bound: 187.6163102
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266363, upper bound: 187.6178248
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6178620, upper bound: 187.6277106
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6198769, upper bound: 187.6165894
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258914, upper bound: 187.6268523
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6294001, upper bound: 187.6165894
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6252971, upper bound: 187.6271693
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6288528, upper bound: 187.6165894
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6254669, upper bound: 187.6270188
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290411, upper bound: 187.6165894
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6285131
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6296686, upper bound: 187.6251096
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6281606
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302788, upper bound: 187.6245421
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169452, upper bound: 187.6295640
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6296988, upper bound: 187.6271599
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=232.61239624023438
rel_dist={3: [-187.90965608592424, 187.9096560859242]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6488867, upper bound: 187.6501757
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6501757, upper bound: 187.6488867
time: 0.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 3, lower bound: -187.6488867, upper bound: 187.6501757
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 3, lower bound: -187.6501757, upper bound: 187.6488867

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6270118, upper bound: 187.6271729
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271729, upper bound: 187.6270118
time: 0.65 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6270118, upper bound: 187.6271729
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6294621, upper bound: 187.6271118
time: 0.64 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.11 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 3, lower bound: -187.6270118, upper bound: 187.6271729
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 3, lower bound: -187.6271729, upper bound: 187.6270118
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 3, lower bound: -187.6270118, upper bound: 187.6271729
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 3, lower bound: -187.6294621, upper bound: 187.6271118

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6270143, upper bound: 187.6294070
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271118, upper bound: 187.6293621
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271729, upper bound: 187.6270118
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271626, upper bound: 187.6241016
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6241016, upper bound: 187.6271626
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6270118, upper bound: 187.6271729
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6293621, upper bound: 187.6271118
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6294070, upper bound: 187.6270143
time: 0.59 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.22 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 3, lower bound: -187.6270143, upper bound: 187.6294070
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 3, lower bound: -187.6271118, upper bound: 187.6293621
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 3, lower bound: -187.6271729, upper bound: 187.6270118
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 3, lower bound: -187.6271626, upper bound: 187.6241016
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 3, lower bound: -187.6241016, upper bound: 187.6271626
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 3, lower bound: -187.6270118, upper bound: 187.6271729
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 3, lower bound: -187.6293621, upper bound: 187.6271118
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 3, lower bound: -187.6294070, upper bound: 187.6270143

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6251547, upper bound: 187.6290191
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268038, upper bound: 187.6263553
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6251853, upper bound: 187.6289478
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269199, upper bound: 187.6263666
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6251712, upper bound: 187.6265564
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269514, upper bound: 187.6252037
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6251853, upper bound: 187.6175169
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269629, upper bound: 187.6192697
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6175169, upper bound: 187.6269629
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6175169, upper bound: 187.6251853
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6252037, upper bound: 187.6269514
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6265564, upper bound: 187.6251712
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263666, upper bound: 187.6269199
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289478, upper bound: 187.6251853
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263553, upper bound: 187.6268038
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290191, upper bound: 187.6251547
time: 0.68 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -187.6251547, upper bound: 187.6290191
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -187.6268038, upper bound: 187.6263553
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -187.6251853, upper bound: 187.6289478
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -187.6269199, upper bound: 187.6263666
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -187.6251712, upper bound: 187.6265564
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -187.6269514, upper bound: 187.6252037
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -187.6251853, upper bound: 187.6175169
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -187.6269629, upper bound: 187.6192697
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -187.6175169, upper bound: 187.6269629
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -187.6175169, upper bound: 187.6251853
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -187.6252037, upper bound: 187.6269514
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -187.6265564, upper bound: 187.6251712
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -187.6263666, upper bound: 187.6269199
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -187.6289478, upper bound: 187.6251853
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -187.6263553, upper bound: 187.6268038
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -187.6290191, upper bound: 187.6251547

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248252, upper bound: 187.6273767
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6245879, upper bound: 187.6284032
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263259, upper bound: 187.6260754
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6260525
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248500, upper bound: 187.6210758
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248769, upper bound: 187.6283132
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6264544, upper bound: 187.6257650
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6255791, upper bound: 187.6260478
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248419, upper bound: 187.6258371
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6246034, upper bound: 187.6261615
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6264737, upper bound: 187.6249857
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6175253
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248602, upper bound: 187.6169241
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248769, upper bound: 187.6169241
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6264975, upper bound: 187.6189098
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6251945, upper bound: 187.6169241
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6251945
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248602
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248769
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248602
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6175253, upper bound: 187.6169241
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6249857, upper bound: 187.6264737
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6261615, upper bound: 187.6246034
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258371, upper bound: 187.6248419
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260478, upper bound: 187.6255791
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6257650, upper bound: 187.6264544
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6283132, upper bound: 187.6248769
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6210758, upper bound: 187.6248500
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260525, upper bound: 187.6169241
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6189098, upper bound: 187.6263259
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6284032, upper bound: 187.6245879
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6273767, upper bound: 187.6248252
time: 0.71 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6248252, upper bound: 187.6273767
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6245879, upper bound: 187.6284032
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6263259, upper bound: 187.6260754
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6260525
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6248500, upper bound: 187.6210758
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6248769, upper bound: 187.6283132
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6264544, upper bound: 187.6257650
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6255791, upper bound: 187.6260478
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6248419, upper bound: 187.6258371
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6246034, upper bound: 187.6261615
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6264737, upper bound: 187.6249857
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6175253
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6248602, upper bound: 187.6169241
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6248769, upper bound: 187.6169241
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6264975, upper bound: 187.6189098
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6251945, upper bound: 187.6169241
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6251945
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248602
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248769
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248602
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6175253, upper bound: 187.6169241
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6249857, upper bound: 187.6264737
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6261615, upper bound: 187.6246034
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6258371, upper bound: 187.6248419
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6260478, upper bound: 187.6255791
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6257650, upper bound: 187.6264544
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6283132, upper bound: 187.6248769
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6210758, upper bound: 187.6248500
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6260525, upper bound: 187.6169241
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6189098, upper bound: 187.6263259
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6284032, upper bound: 187.6245879
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.29
Output dim: 3, lower bound: -187.6273767, upper bound: 187.6248252

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6247135, upper bound: 187.6273767
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248252, upper bound: 187.6271168
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6245879, upper bound: 187.6284032
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6255575
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6260560
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263259, upper bound: 187.6260754
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6260525
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6257288
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248500, upper bound: 187.6210758
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248427, upper bound: 187.6169241
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248769, upper bound: 187.6283132
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6250008, upper bound: 187.6257650
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6264544, upper bound: 187.6257133
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6253491, upper bound: 187.6260478
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6255791, upper bound: 187.6257043
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6247286, upper bound: 187.6258371
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248419, upper bound: 187.6256914
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6246034, upper bound: 187.6261615
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6201138
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6264737, upper bound: 187.6249798
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6175253
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248602, upper bound: 187.6169241
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248554, upper bound: 187.6169241
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248769, upper bound: 187.6169241
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6250072, upper bound: 187.6169241
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6264975, upper bound: 187.6189098
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6251289, upper bound: 187.6169241
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6251913, upper bound: 187.6169241
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6251913
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6251289
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248554
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6250072
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248769
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248554
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248602
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6175253, upper bound: 187.6169241
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6249798, upper bound: 187.6264737
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6201138, upper bound: 187.6169241
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6261615, upper bound: 187.6246034
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6256914, upper bound: 187.6248419
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258371, upper bound: 187.6247286
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6257043, upper bound: 187.6255791
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260478, upper bound: 187.6253491
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6257133, upper bound: 187.6264544
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6257650, upper bound: 187.6250008
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6283132, upper bound: 187.6248769
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248427
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6210758, upper bound: 187.6248500
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6257288, upper bound: 187.6169241
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260525, upper bound: 187.6169241
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260754, upper bound: 187.6263259
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260560, upper bound: 187.6169241
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6255575, upper bound: 187.6169241
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6284032, upper bound: 187.6245879
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271168, upper bound: 187.6248252
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6273767, upper bound: 187.6247135
time: 0.59 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.72 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6247135, upper bound: 187.6273767
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6248252, upper bound: 187.6271168
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6245879, upper bound: 187.6284032
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6255575
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6260560
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6263259, upper bound: 187.6260754
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6260525
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6257288
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6248500, upper bound: 187.6210758
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6248427, upper bound: 187.6169241
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6248769, upper bound: 187.6283132
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6250008, upper bound: 187.6257650
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6264544, upper bound: 187.6257133
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6253491, upper bound: 187.6260478
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6255791, upper bound: 187.6257043
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6247286, upper bound: 187.6258371
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6248419, upper bound: 187.6256914
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6246034, upper bound: 187.6261615
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6201138
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6264737, upper bound: 187.6249798
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6175253
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6248602, upper bound: 187.6169241
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6248554, upper bound: 187.6169241
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6248769, upper bound: 187.6169241
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6250072, upper bound: 187.6169241
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6264975, upper bound: 187.6189098
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6251289, upper bound: 187.6169241
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6251913, upper bound: 187.6169241
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6251913
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6251289
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248554
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6250072
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248769
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248554
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248602
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6175253, upper bound: 187.6169241
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6249798, upper bound: 187.6264737
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6201138, upper bound: 187.6169241
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6261615, upper bound: 187.6246034
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6256914, upper bound: 187.6248419
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6258371, upper bound: 187.6247286
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6257043, upper bound: 187.6255791
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6260478, upper bound: 187.6253491
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6257133, upper bound: 187.6264544
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6257650, upper bound: 187.6250008
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6283132, upper bound: 187.6248769
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248427
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6210758, upper bound: 187.6248500
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6257288, upper bound: 187.6169241
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6260525, upper bound: 187.6169241
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6260754, upper bound: 187.6263259
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6260560, upper bound: 187.6169241
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6255575, upper bound: 187.6169241
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6284032, upper bound: 187.6245879
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6271168, upper bound: 187.6248252
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -187.6273767, upper bound: 187.6247135

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6273767
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6247135, upper bound: 187.6237408
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6271168
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248252, upper bound: 187.6235635
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6284032
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6245879, upper bound: 187.6240563
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6255575
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6165686
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6260560
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6260754
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263259, upper bound: 187.6203925
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6260525
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6257288
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6241782, upper bound: 187.6174059
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168181, upper bound: 187.6203616
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6165686
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248427, upper bound: 187.6165686
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6241885, upper bound: 187.6181466
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6170727, upper bound: 187.6278510
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6165686
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6165686
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6257650
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6250008, upper bound: 187.6169241
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6244689, upper bound: 187.6257133
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6264544, upper bound: 187.6169241
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6260478
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6253491, upper bound: 187.6169241
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6228465, upper bound: 187.6257043
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6255791, upper bound: 187.6169241
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6258371
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6247286, upper bound: 187.6226360
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6256914
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248419, upper bound: 187.6224700
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6261615
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6246034, upper bound: 187.6230668
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6198467
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6165686
time: 0.71 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6273767
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6247135, upper bound: 187.6237408
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6271168
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6248252, upper bound: 187.6235635
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6284032
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6245879, upper bound: 187.6240563
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6255575
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6165686
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6260560
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6260754
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6263259, upper bound: 187.6203925
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6260525
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6257288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6241782, upper bound: 187.6174059
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6168181, upper bound: 187.6203616
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6165686
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6248427, upper bound: 187.6165686
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6241885, upper bound: 187.6181466
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6170727, upper bound: 187.6278510
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6165686
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6165686
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6257650
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6250008, upper bound: 187.6169241
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6244689, upper bound: 187.6257133
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6264544, upper bound: 187.6169241
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6260478
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6253491, upper bound: 187.6169241
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6228465, upper bound: 187.6257043
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6255791, upper bound: 187.6169241
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6258371
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6247286, upper bound: 187.6226360
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6256914
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6248419, upper bound: 187.6224700
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6261615
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6246034, upper bound: 187.6230668
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6198467
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 3, lower bound: -187.6165686, upper bound: 187.6165686
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6264737, upper bound: 187.6249798
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6175253
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6248602, upper bound: 187.6169241
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6248554, upper bound: 187.6169241
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6248769, upper bound: 187.6169241
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6250072, upper bound: 187.6169241
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6264975, upper bound: 187.6189098
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6251289, upper bound: 187.6169241
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6251913, upper bound: 187.6169241
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6251913
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6251289
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248554
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6250072
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248769
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248554
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248602
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6175253, upper bound: 187.6169241
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6249798, upper bound: 187.6264737
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6201138, upper bound: 187.6169241
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6261615, upper bound: 187.6246034
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6256914, upper bound: 187.6248419
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6258371, upper bound: 187.6247286
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6257043, upper bound: 187.6255791
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6260478, upper bound: 187.6253491
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6257133, upper bound: 187.6264544
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6257650, upper bound: 187.6250008
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6169241
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6283132, upper bound: 187.6248769
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6169241, upper bound: 187.6248427
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6210758, upper bound: 187.6248500
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6257288, upper bound: 187.6169241
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6260525, upper bound: 187.6169241
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6260754, upper bound: 187.6263259
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6260560, upper bound: 187.6169241
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6255575, upper bound: 187.6169241
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6284032, upper bound: 187.6245879
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6271168, upper bound: 187.6248252
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 3, lower bound: -187.6273767, upper bound: 187.6247135
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=232.61239624023438
rel_dist={3: [-187.89872093335524, 187.8987209333552]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1137.99 seconds
