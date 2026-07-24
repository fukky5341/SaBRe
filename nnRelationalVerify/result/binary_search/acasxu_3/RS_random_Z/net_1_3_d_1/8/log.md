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
execution time: IAR + LP analysis = 1.74 + 1.73 = 3.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -187.9182065, upper bound: 187.9182065


# Binary Search by BASE starts (time budget: 1196.53 seconds, max iter: 100)

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
Binary search time: 59.05 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1137.48 seconds

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8944562, upper bound: 187.9182065
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.9182065, upper bound: 187.8944562
time: 0.56 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.18 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 3, lower bound: -187.8944562, upper bound: 187.9182065
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 3, lower bound: -187.9182065, upper bound: 187.8944562

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8902419, upper bound: 187.9153184
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8918266, upper bound: 187.9154886
time: 0.84 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.9164790, upper bound: 187.8944562
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.9182065, upper bound: 187.8943707
time: 0.65 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.96 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 3, lower bound: -187.8902419, upper bound: 187.9153184
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 3, lower bound: -187.8918266, upper bound: 187.9154886
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 3, lower bound: -187.9164790, upper bound: 187.8944562
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 3, lower bound: -187.9182065, upper bound: 187.8943707

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6672817, upper bound: 187.6774457
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6672817, upper bound: 187.6721943
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8472949, upper bound: 187.8486046
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8472949, upper bound: 187.8486046
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6535012, upper bound: 187.6489994
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6581434, upper bound: 187.6489994
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.9105499, upper bound: 187.8888596
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.9178349, upper bound: 187.8833960
time: 0.68 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.12 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 3, lower bound: -187.6672817, upper bound: 187.6774457
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 3, lower bound: -187.6672817, upper bound: 187.6721943
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 3, lower bound: -187.8472949, upper bound: 187.8486046
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 3, lower bound: -187.8472949, upper bound: 187.8486046
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 3, lower bound: -187.6535012, upper bound: 187.6489994
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 3, lower bound: -187.6581434, upper bound: 187.6489994
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 3, lower bound: -187.9105499, upper bound: 187.8888596
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 3, lower bound: -187.9178349, upper bound: 187.8833960

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6657710, upper bound: 187.6764939
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6632769, upper bound: 187.6766254
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6659651, upper bound: 187.6721943
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6672688, upper bound: 187.6679395
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7947119, upper bound: 187.7952361
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7947119, upper bound: 187.7950187
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8462712, upper bound: 187.8432653
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8453720, upper bound: 187.8469043
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6513039, upper bound: 187.6453973
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6456590, upper bound: 187.6477451
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5888729, upper bound: 187.5861471
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5887945, upper bound: 187.5861471
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8360267, upper bound: 187.8417386
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8360267, upper bound: 187.8417386
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8469367, upper bound: 187.8193935
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8469367, upper bound: 187.8193935
time: 0.61 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 3, lower bound: -187.6657710, upper bound: 187.6764939
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 3, lower bound: -187.6632769, upper bound: 187.6766254
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 3, lower bound: -187.6659651, upper bound: 187.6721943
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 3, lower bound: -187.6672688, upper bound: 187.6679395
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 3, lower bound: -187.7947119, upper bound: 187.7952361
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 3, lower bound: -187.7947119, upper bound: 187.7950187
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 3, lower bound: -187.8462712, upper bound: 187.8432653
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 3, lower bound: -187.8453720, upper bound: 187.8469043
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 3, lower bound: -187.6513039, upper bound: 187.6453973
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 3, lower bound: -187.6456590, upper bound: 187.6477451
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 3, lower bound: -187.5888729, upper bound: 187.5861471
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 3, lower bound: -187.5887945, upper bound: 187.5861471
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 3, lower bound: -187.8360267, upper bound: 187.8417386
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 3, lower bound: -187.8360267, upper bound: 187.8417386
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 3, lower bound: -187.8469367, upper bound: 187.8193935
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 3, lower bound: -187.8469367, upper bound: 187.8193935

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6639279, upper bound: 187.6764939
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6657234, upper bound: 187.6701202
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6631801, upper bound: 187.6766254
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6631801, upper bound: 187.6689448
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6646926, upper bound: 187.6642770
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6630117, upper bound: 187.6697118
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6591387, upper bound: 187.6651734
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6672564, upper bound: 187.6639211
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7934221, upper bound: 187.7934221
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7934221, upper bound: 187.7934461
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7934221, upper bound: 187.7934221
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7934221, upper bound: 187.7934230
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8394546, upper bound: 187.8364032
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8377706, upper bound: 187.8364032
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8447739, upper bound: 187.8469043
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8453720, upper bound: 187.8432653
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6511458, upper bound: 187.6453686
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6513039, upper bound: 187.6448989
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6408159, upper bound: 187.6474428
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6408676, upper bound: 187.6400434
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5395076, upper bound: 187.5395076
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5420869, upper bound: 187.5405368
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5395076, upper bound: 187.5395076
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5416496, upper bound: 187.5405368
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7917436, upper bound: 187.7917436
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7917436, upper bound: 187.7917436
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7917436, upper bound: 187.7917436
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7917436, upper bound: 187.7917436
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8409890, upper bound: 187.8144872
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8221026, upper bound: 187.8140746
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5941933, upper bound: 187.5861471
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5962609, upper bound: 187.5861471
time: 0.75 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.09 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.6639279, upper bound: 187.6764939
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.6657234, upper bound: 187.6701202
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.6631801, upper bound: 187.6766254
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.6631801, upper bound: 187.6689448
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.6646926, upper bound: 187.6642770
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.6630117, upper bound: 187.6697118
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.6591387, upper bound: 187.6651734
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.6672564, upper bound: 187.6639211
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.7934221, upper bound: 187.7934221
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.7934221, upper bound: 187.7934461
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.7934221, upper bound: 187.7934221
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.7934221, upper bound: 187.7934230
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.8394546, upper bound: 187.8364032
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.8377706, upper bound: 187.8364032
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.8447739, upper bound: 187.8469043
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.8453720, upper bound: 187.8432653
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.6511458, upper bound: 187.6453686
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.6513039, upper bound: 187.6448989
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.6408159, upper bound: 187.6474428
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.6408676, upper bound: 187.6400434
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.5395076, upper bound: 187.5395076
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.5420869, upper bound: 187.5405368
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.5395076, upper bound: 187.5395076
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.5416496, upper bound: 187.5405368
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.7917436, upper bound: 187.7917436
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.7917436, upper bound: 187.7917436
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.7917436, upper bound: 187.7917436
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.7917436, upper bound: 187.7917436
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.8409890, upper bound: 187.8144872
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.8221026, upper bound: 187.8140746
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.5941933, upper bound: 187.5861471
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 3, lower bound: -187.5962609, upper bound: 187.5861471

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6631801, upper bound: 187.6764939
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6639279, upper bound: 187.6763972
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6645174, upper bound: 187.6642404
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6618836, upper bound: 187.6691375
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6578392, upper bound: 187.6761199
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6578392, upper bound: 187.6725264
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6578392, upper bound: 187.6669802
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6578392, upper bound: 187.6578392
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6264207, upper bound: 187.6252059
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6281359, upper bound: 187.6246414
time: 0.65 seconds

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
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6068423
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6069984
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6591387, upper bound: 187.6596647
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6591387, upper bound: 187.6651734
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6286783, upper bound: 187.6248319
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6293327, upper bound: 187.6227426
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7859102
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7844749, upper bound: 187.7844749
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7844749, upper bound: 187.7844749
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7859162, upper bound: 187.7907896
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7859162, upper bound: 187.7859162
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8330860, upper bound: 187.8297889
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8308128, upper bound: 187.8297889
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8364032, upper bound: 187.8364032
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8377706, upper bound: 187.8364032
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8377444, upper bound: 187.8400987
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8364034, upper bound: 187.8400392
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8387390, upper bound: 187.8363178
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8370922, upper bound: 187.8363178
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291499, upper bound: 187.6255131
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6292754, upper bound: 187.6250419
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5810130, upper bound: 187.5810130
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5810130, upper bound: 187.5810130
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6404386, upper bound: 187.6474027
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6404386, upper bound: 187.6462158
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5861471, upper bound: 187.5861471
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5861471, upper bound: 187.5861471
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7872744, upper bound: 187.7872744
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7872744, upper bound: 187.7872744
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7872744, upper bound: 187.7872744
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7872744, upper bound: 187.7872744
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7903859, upper bound: 187.7903859
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7903859, upper bound: 187.7903859
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7825911, upper bound: 187.7825911
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7825911, upper bound: 187.7825911
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5889571, upper bound: 187.5810130
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5903790, upper bound: 187.5810130
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6134532, upper bound: 187.6058815
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6134532, upper bound: 187.6058815
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5889571, upper bound: 187.5810130
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5874856, upper bound: 187.5810130
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5831513, upper bound: 187.5829732
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5926502, upper bound: 187.5829732
time: 0.72 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6631801, upper bound: 187.6764939
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6639279, upper bound: 187.6763972
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6645174, upper bound: 187.6642404
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6618836, upper bound: 187.6691375
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6578392, upper bound: 187.6761199
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6578392, upper bound: 187.6725264
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6578392, upper bound: 187.6669802
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6578392, upper bound: 187.6578392
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6264207, upper bound: 187.6252059
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6281359, upper bound: 187.6246414
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6068423
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6069984
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6591387, upper bound: 187.6596647
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6591387, upper bound: 187.6651734
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6286783, upper bound: 187.6248319
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6293327, upper bound: 187.6227426
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7859102
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.7844749, upper bound: 187.7844749
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.7844749, upper bound: 187.7844749
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.7859162, upper bound: 187.7907896
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.7859162, upper bound: 187.7859162
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.8330860, upper bound: 187.8297889
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.8308128, upper bound: 187.8297889
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.8364032, upper bound: 187.8364032
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.8377706, upper bound: 187.8364032
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.8377444, upper bound: 187.8400987
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.8364034, upper bound: 187.8400392
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.8387390, upper bound: 187.8363178
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.8370922, upper bound: 187.8363178
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6291499, upper bound: 187.6255131
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6292754, upper bound: 187.6250419
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.5810130, upper bound: 187.5810130
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.5810130, upper bound: 187.5810130
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6404386, upper bound: 187.6474027
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6404386, upper bound: 187.6462158
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.5861471, upper bound: 187.5861471
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.5861471, upper bound: 187.5861471
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.7872744, upper bound: 187.7872744
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.7872744, upper bound: 187.7872744
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.7872744, upper bound: 187.7872744
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.7872744, upper bound: 187.7872744
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.7903859, upper bound: 187.7903859
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.7903859, upper bound: 187.7903859
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.7825911, upper bound: 187.7825911
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.7825911, upper bound: 187.7825911
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.5889571, upper bound: 187.5810130
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.5903790, upper bound: 187.5810130
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6134532, upper bound: 187.6058815
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6134532, upper bound: 187.6058815
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.5889571, upper bound: 187.5810130
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.5874856, upper bound: 187.5810130
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.5831513, upper bound: 187.5829732
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.5926502, upper bound: 187.5829732

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6245144, upper bound: 187.6384940
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6245144, upper bound: 187.6307276
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263704, upper bound: 187.6382909
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6245144, upper bound: 187.6307298
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6618836, upper bound: 187.6625047
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6645174, upper bound: 187.6642404
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6618836, upper bound: 187.6685715
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6620143, upper bound: 187.6691375
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6383088
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6300938
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6578392, upper bound: 187.6725264
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6578392, upper bound: 187.6718154
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6579296
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6660867
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5283737, upper bound: 187.5283737
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5283737, upper bound: 187.5283737
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6243250, upper bound: 187.6245904
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6281359, upper bound: 187.6246414
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6068423
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6036105
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5283737, upper bound: 187.5283737
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5283737, upper bound: 187.5349357
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6178080, upper bound: 187.6193399
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6178080, upper bound: 187.6178080
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6178080, upper bound: 187.6265819
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6178080, upper bound: 187.6248629
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6276435, upper bound: 187.6182916
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6185860, upper bound: 187.6239530
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5293046, upper bound: 187.5293046
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5293046, upper bound: 187.5293046
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7771515, upper bound: 187.7771515
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7771515, upper bound: 187.7771515
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7771515, upper bound: 187.7771692
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7771515, upper bound: 187.7771603
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7771515, upper bound: 187.7771515
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7771515, upper bound: 187.7771515
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7768119, upper bound: 187.7768119
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7768325, upper bound: 187.7768119
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7801181
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7832537
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7768119, upper bound: 187.7768119
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7768119, upper bound: 187.7768119
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8297889, upper bound: 187.8297889
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8330860, upper bound: 187.8297889
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7771515, upper bound: 187.7771515
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7771515, upper bound: 187.7771515
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8298285, upper bound: 187.8297889
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8297889, upper bound: 187.8300436
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8312442, upper bound: 187.8297889
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8302113, upper bound: 187.8299689
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7844749, upper bound: 187.7844926
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7844749, upper bound: 187.7844749
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8299004, upper bound: 187.8308689
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8297889, upper bound: 187.8337001
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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8319696, upper bound: 187.8297889
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8299042, upper bound: 187.8297889
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8251844
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8251844
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5346237
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5346237
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6243250, upper bound: 187.6247009
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6292754, upper bound: 187.6248837
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5404658, upper bound: 187.5346237
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5357933, upper bound: 187.5346237
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

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
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5792582, upper bound: 187.5775126
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6384472, upper bound: 187.6439010
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6384472, upper bound: 187.6460580
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6384472, upper bound: 187.6435989
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6393688, upper bound: 187.6447974
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5394760, upper bound: 187.5391030
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5391030, upper bound: 187.5391030
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5810130, upper bound: 187.5810130
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5810130, upper bound: 187.5810130
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7859162, upper bound: 187.7859162
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7859162, upper bound: 187.7859162
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7859162, upper bound: 187.7859162
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7859162, upper bound: 187.7859162
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7797385, upper bound: 187.7797385
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7797385, upper bound: 187.7797385
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7859162, upper bound: 187.7859162
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7859162, upper bound: 187.7859162
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7781939, upper bound: 187.7781939
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7781939, upper bound: 187.7781939
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7781939, upper bound: 187.7781939
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7781939, upper bound: 187.7781939
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5437233, upper bound: 187.5361135
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5438093, upper bound: 187.5361135
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5812829, upper bound: 187.5810130
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5812829, upper bound: 187.5810130
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6014753
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6090491, upper bound: 187.6014753
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6014753
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6090491, upper bound: 187.6014753
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5850528, upper bound: 187.5775126
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5836253, upper bound: 187.5775126
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775214, upper bound: 187.5775126
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5864747, upper bound: 187.5775126
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5836253, upper bound: 187.5775126
time: 0.65 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6245144, upper bound: 187.6384940
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6245144, upper bound: 187.6307276
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6263704, upper bound: 187.6382909
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6245144, upper bound: 187.6307298
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6618836, upper bound: 187.6625047
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6645174, upper bound: 187.6642404
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6618836, upper bound: 187.6685715
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6620143, upper bound: 187.6691375
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6383088
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6300938
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6578392, upper bound: 187.6725264
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6578392, upper bound: 187.6718154
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6579296
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6660867
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5283737, upper bound: 187.5283737
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5283737, upper bound: 187.5283737
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6243250, upper bound: 187.6245904
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6281359, upper bound: 187.6246414
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6068423
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6036105
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5283737, upper bound: 187.5283737
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5283737, upper bound: 187.5349357
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6178080, upper bound: 187.6193399
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6178080, upper bound: 187.6178080
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6178080, upper bound: 187.6265819
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6178080, upper bound: 187.6248629
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6276435, upper bound: 187.6182916
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6185860, upper bound: 187.6239530
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5293046, upper bound: 187.5293046
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5293046, upper bound: 187.5293046
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7771515, upper bound: 187.7771515
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7771515, upper bound: 187.7771515
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7771515, upper bound: 187.7771692
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7771515, upper bound: 187.7771603
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7771515, upper bound: 187.7771515
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7771515, upper bound: 187.7771515
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7768119, upper bound: 187.7768119
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7768325, upper bound: 187.7768119
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7801181
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7832537
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7768119, upper bound: 187.7768119
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7768119, upper bound: 187.7768119
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.8297889, upper bound: 187.8297889
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.8330860, upper bound: 187.8297889
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7771515, upper bound: 187.7771515
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7771515, upper bound: 187.7771515
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.8298285, upper bound: 187.8297889
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.8297889, upper bound: 187.8300436
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.8312442, upper bound: 187.8297889
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.8302113, upper bound: 187.8299689
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7844749, upper bound: 187.7844926
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7844749, upper bound: 187.7844749
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.8299004, upper bound: 187.8308689
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.8297889, upper bound: 187.8337001
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.8319696, upper bound: 187.8297889
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.8299042, upper bound: 187.8297889
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8251844
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8251844
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5346237
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5346237
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6243250, upper bound: 187.6247009
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6292754, upper bound: 187.6248837
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5404658, upper bound: 187.5346237
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5357933, upper bound: 187.5346237
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5792582, upper bound: 187.5775126
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6384472, upper bound: 187.6439010
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6384472, upper bound: 187.6460580
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6384472, upper bound: 187.6435989
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6393688, upper bound: 187.6447974
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5394760, upper bound: 187.5391030
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5391030, upper bound: 187.5391030
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5810130, upper bound: 187.5810130
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5810130, upper bound: 187.5810130
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7859162, upper bound: 187.7859162
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7859162, upper bound: 187.7859162
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7859162, upper bound: 187.7859162
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7859162, upper bound: 187.7859162
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7797385, upper bound: 187.7797385
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7797385, upper bound: 187.7797385
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7859162, upper bound: 187.7859162
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7859162, upper bound: 187.7859162
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7781939, upper bound: 187.7781939
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7781939, upper bound: 187.7781939
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7781939, upper bound: 187.7781939
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.7781939, upper bound: 187.7781939
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5437233, upper bound: 187.5361135
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5438093, upper bound: 187.5361135
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5812829, upper bound: 187.5810130
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5812829, upper bound: 187.5810130
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6014753
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6090491, upper bound: 187.6014753
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6014753
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.6090491, upper bound: 187.6014753
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5850528, upper bound: 187.5775126
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5836253, upper bound: 187.5775126
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5775214, upper bound: 187.5775126
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5864747, upper bound: 187.5775126
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -187.5836253, upper bound: 187.5775126

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6236061, upper bound: 187.6242878
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6236061, upper bound: 187.6350663
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6289107
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6165894
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6253213, upper bound: 187.6253922
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6236061, upper bound: 187.6350386
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6236061, upper bound: 187.6239054
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6236061, upper bound: 187.6289947
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6236061, upper bound: 187.6244544
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6236061, upper bound: 187.6236061
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6580841
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6645174, upper bound: 187.6581652
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6236061, upper bound: 187.6320079
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6236061, upper bound: 187.6236061
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6649297
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6574775, upper bound: 187.6632506
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6173597
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6324002
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6300938
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6284335
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6324057
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6280248
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.95 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=232.61239624023438
rel_dist={3: [-187.91820645300623, 187.91820645300623]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8439725, upper bound: 187.8439725
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8439725, upper bound: 187.8439725
time: 0.62 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.27 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.27
Output dim: 3, lower bound: -187.8439725, upper bound: 187.8439725
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.27
Output dim: 3, lower bound: -187.8439725, upper bound: 187.8439725

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5940211, upper bound: 187.5950984
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5950984, upper bound: 187.5940211
time: 0.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8420102, upper bound: 187.8280004
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8280004, upper bound: 187.8420102
time: 0.60 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.04 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.04
Output dim: 3, lower bound: -187.5940211, upper bound: 187.5950984
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.04
Output dim: 3, lower bound: -187.5950984, upper bound: 187.5940211
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.04
Output dim: 3, lower bound: -187.8420102, upper bound: 187.8280004
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.04
Output dim: 3, lower bound: -187.8280004, upper bound: 187.8420102

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5861471, upper bound: 187.5950984
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5940211, upper bound: 187.5861471
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5924582, upper bound: 187.5865053
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5865053, upper bound: 187.5923737
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6156296, upper bound: 187.6093999
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6158709, upper bound: 187.6093999
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8265552, upper bound: 187.8255017
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8229427, upper bound: 187.8308066
time: 0.71 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.14 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 3, lower bound: -187.5861471, upper bound: 187.5950984
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 3, lower bound: -187.5940211, upper bound: 187.5861471
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 3, lower bound: -187.5924582, upper bound: 187.5865053
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 3, lower bound: -187.5865053, upper bound: 187.5923737
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 3, lower bound: -187.6156296, upper bound: 187.6093999
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 3, lower bound: -187.6158709, upper bound: 187.6093999
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 3, lower bound: -187.8265552, upper bound: 187.8255017
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 3, lower bound: -187.8229427, upper bound: 187.8308066

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5407102, upper bound: 187.5492713
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5395076, upper bound: 187.5395076
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5829732, upper bound: 187.5829732
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5904649, upper bound: 187.5829732
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5867199, upper bound: 187.5810195
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5862413, upper bound: 187.5812829
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5863953, upper bound: 187.5923737
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5865053, upper bound: 187.5903037
time: 0.56 seconds

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6134777, upper bound: 187.6093999
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6156296, upper bound: 187.6092883
time: 0.55 seconds

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
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6134777, upper bound: 187.6093999
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6158709, upper bound: 187.6092883
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8214725, upper bound: 187.8207084
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8182121, upper bound: 187.8180204
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8222177, upper bound: 187.8299437
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8222774, upper bound: 187.8222177
time: 0.57 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.46 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 3, lower bound: -187.5407102, upper bound: 187.5492713
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.46
Output dim: 3, lower bound: -187.5395076, upper bound: 187.5395076
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 3, lower bound: -187.5829732, upper bound: 187.5829732
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 3, lower bound: -187.5904649, upper bound: 187.5829732
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 3, lower bound: -187.5867199, upper bound: 187.5810195
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 3, lower bound: -187.5862413, upper bound: 187.5812829
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 3, lower bound: -187.5863953, upper bound: 187.5923737
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 3, lower bound: -187.5865053, upper bound: 187.5903037
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 3, lower bound: -187.6134777, upper bound: 187.6093999
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 3, lower bound: -187.6156296, upper bound: 187.6092883
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 3, lower bound: -187.6134777, upper bound: 187.6093999
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 3, lower bound: -187.6158709, upper bound: 187.6092883
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 3, lower bound: -187.8214725, upper bound: 187.8207084
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 3, lower bound: -187.8182121, upper bound: 187.8180204
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 3, lower bound: -187.8222177, upper bound: 187.8299437
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 3, lower bound: -187.8222774, upper bound: 187.8222177

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5407102, upper bound: 187.5492713
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5406272, upper bound: 187.5432420
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5829732, upper bound: 187.5829732
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5829732, upper bound: 187.5829732
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5422712, upper bound: 187.5347452
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5423366, upper bound: 187.5347452
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5346237
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5414014, upper bound: 187.5348825
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5810130, upper bound: 187.5812829
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5862413, upper bound: 187.5810130
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5829732, upper bound: 187.5887630
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5832082, upper bound: 187.5829732
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5391030, upper bound: 187.5428982
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5394737, upper bound: 187.5449093
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6090473, upper bound: 187.6093999
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6134777, upper bound: 187.6090473
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6048332
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6111074, upper bound: 187.6045916
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5428982, upper bound: 187.5391030
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5428196, upper bound: 187.5394816
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6090473, upper bound: 187.6092883
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6158709, upper bound: 187.6091791
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8130016, upper bound: 187.8169463
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8172059, upper bound: 187.8185420
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8077857, upper bound: 187.8077857
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8080171, upper bound: 187.8077857
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7897070, upper bound: 187.7904284
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7897070, upper bound: 187.7904284
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7897070, upper bound: 187.7902006
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7897070, upper bound: 187.7901380
time: 0.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.5407102, upper bound: 187.5492713
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.5406272, upper bound: 187.5432420
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.5829732, upper bound: 187.5829732
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.5829732, upper bound: 187.5829732
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.5422712, upper bound: 187.5347452
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.5423366, upper bound: 187.5347452
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5346237
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.5414014, upper bound: 187.5348825
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.5810130, upper bound: 187.5812829
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.5862413, upper bound: 187.5810130
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.5829732, upper bound: 187.5887630
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.5832082, upper bound: 187.5829732
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.5391030, upper bound: 187.5428982
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.5394737, upper bound: 187.5449093
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.6090473, upper bound: 187.6093999
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.6134777, upper bound: 187.6090473
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6048332
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.6111074, upper bound: 187.6045916
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.5428982, upper bound: 187.5391030
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.5428196, upper bound: 187.5394816
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.6090473, upper bound: 187.6092883
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.6158709, upper bound: 187.6091791
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.8130016, upper bound: 187.8169463
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.8172059, upper bound: 187.8185420
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.8077857, upper bound: 187.8077857
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.8080171, upper bound: 187.8077857
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.7897070, upper bound: 187.7904284
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.7897070, upper bound: 187.7904284
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.7897070, upper bound: 187.7902006
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -187.7897070, upper bound: 187.7901380

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5406797, upper bound: 187.5394737
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5396082, upper bound: 187.5462968
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5406272, upper bound: 187.5432420
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5405368, upper bound: 187.5416496
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5777814
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5843902, upper bound: 187.5810130
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5862413, upper bound: 187.5810130
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5409574
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5409285
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5830632, upper bound: 187.5829732
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5829732, upper bound: 187.5829732
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5391030, upper bound: 187.5428982
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5391030, upper bound: 187.5411997
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5391030, upper bound: 187.5449093
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5394737, upper bound: 187.5448410
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6049453
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6045993
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5449093, upper bound: 187.5391030
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5391030, upper bound: 187.5391030
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6015862
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6015995
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5409285, upper bound: 187.5343558
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5373189, upper bound: 187.5343558
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5360663, upper bound: 187.5348825
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5371750, upper bound: 187.5348833
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6048332
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6045916
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6048790, upper bound: 187.6047255
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6113487, upper bound: 187.6045916
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8120837, upper bound: 187.8163664
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8123990, upper bound: 187.8130562
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8120837, upper bound: 187.8179610
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8165486, upper bound: 187.8128714
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8024283, upper bound: 187.8024283
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8024283, upper bound: 187.8024283
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8028195, upper bound: 187.8024283
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8024283, upper bound: 187.8024283
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7897070, upper bound: 187.7904284
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7897070, upper bound: 187.7897070
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7879154
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7875018
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7897070, upper bound: 187.7901380
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7897070, upper bound: 187.7897070
time: 0.61 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5406797, upper bound: 187.5394737
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5396082, upper bound: 187.5462968
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5406272, upper bound: 187.5432420
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5405368, upper bound: 187.5416496
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5777814
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5843902, upper bound: 187.5810130
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5862413, upper bound: 187.5810130
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5409574
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5409285
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5830632, upper bound: 187.5829732
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5829732, upper bound: 187.5829732
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5391030, upper bound: 187.5428982
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5391030, upper bound: 187.5411997
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5391030, upper bound: 187.5449093
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5394737, upper bound: 187.5448410
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6049453
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6045993
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5449093, upper bound: 187.5391030
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5391030, upper bound: 187.5391030
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6015862
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6015995
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5409285, upper bound: 187.5343558
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5373189, upper bound: 187.5343558
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5360663, upper bound: 187.5348825
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.5371750, upper bound: 187.5348833
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6048332
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6045916
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.6048790, upper bound: 187.6047255
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.6113487, upper bound: 187.6045916
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.8120837, upper bound: 187.8163664
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.8123990, upper bound: 187.8130562
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.8120837, upper bound: 187.8179610
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.8165486, upper bound: 187.8128714
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.8024283, upper bound: 187.8024283
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.8024283, upper bound: 187.8024283
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.8028195, upper bound: 187.8024283
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.8024283, upper bound: 187.8024283
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.7897070, upper bound: 187.7904284
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.7897070, upper bound: 187.7897070
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7879154
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7875018
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.7897070, upper bound: 187.7901380
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 3, lower bound: -187.7897070, upper bound: 187.7897070

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5396082, upper bound: 187.5462968
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5391030, upper bound: 187.5391030
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5361135, upper bound: 187.5390345
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5365527, upper bound: 187.5381476
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5293046, upper bound: 187.5293046
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5293046, upper bound: 187.5293046
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5293046, upper bound: 187.5293046
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5293046, upper bound: 187.5293046
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5293046, upper bound: 187.5293046
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5293046, upper bound: 187.5293046
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5777814
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5806471, upper bound: 187.5775126
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5346237
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5404751, upper bound: 187.5346237
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5829732, upper bound: 187.5829732
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5832082, upper bound: 187.5829732
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5373189
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5405538
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5351136
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5348880, upper bound: 187.5404658
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5346853, upper bound: 187.5346237
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6049453
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6045916
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6045993
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6045916
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5402013, upper bound: 187.5343558
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6015862
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6014753
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6015995
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6014753
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5347562
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6045916
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6045916
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6047255
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6048790, upper bound: 187.6045916
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5409574, upper bound: 187.5343558
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5416101, upper bound: 187.5343558
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8121285, upper bound: 187.8120837
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8123990, upper bound: 187.8130562
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7710820, upper bound: 187.7710820
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7710820, upper bound: 187.7710820
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7710820, upper bound: 187.7710820
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7710820, upper bound: 187.7710820
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7710820, upper bound: 187.7710820
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7710820, upper bound: 187.7710820
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7710820, upper bound: 187.7710820
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7710820, upper bound: 187.7710820
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7851767, upper bound: 187.7851767
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7851767, upper bound: 187.7857941
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7851767, upper bound: 187.7851767
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7851767, upper bound: 187.7851767
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7879154
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7740584, upper bound: 187.7740584
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7740584, upper bound: 187.7740584
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7875018
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7851767, upper bound: 187.7851767
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7851767, upper bound: 187.7855504
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
time: 0.68 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.33 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5396082, upper bound: 187.5462968
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5391030, upper bound: 187.5391030
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5361135, upper bound: 187.5390345
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5365527, upper bound: 187.5381476
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5293046, upper bound: 187.5293046
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5293046, upper bound: 187.5293046
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5293046, upper bound: 187.5293046
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5293046, upper bound: 187.5293046
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5293046, upper bound: 187.5293046
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5293046, upper bound: 187.5293046
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5777814
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5806471, upper bound: 187.5775126
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5346237
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5404751, upper bound: 187.5346237
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5829732, upper bound: 187.5829732
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5832082, upper bound: 187.5829732
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5373189
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5405538
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5351136
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5348880, upper bound: 187.5404658
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5346853, upper bound: 187.5346237
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6049453
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6045916
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6045993
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6045916
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5402013, upper bound: 187.5343558
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6015862
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6014753
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6015995
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6014753
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5347562
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6045916
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6045916
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.6045916, upper bound: 187.6047255
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.6048790, upper bound: 187.6045916
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5409574, upper bound: 187.5343558
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.5416101, upper bound: 187.5343558
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.8121285, upper bound: 187.8120837
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.8123990, upper bound: 187.8130562
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7710820, upper bound: 187.7710820
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7710820, upper bound: 187.7710820
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7710820, upper bound: 187.7710820
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7710820, upper bound: 187.7710820
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7710820, upper bound: 187.7710820
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7710820, upper bound: 187.7710820
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7710820, upper bound: 187.7710820
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7710820, upper bound: 187.7710820
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7851767, upper bound: 187.7851767
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7851767, upper bound: 187.7857941
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7851767, upper bound: 187.7851767
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7851767, upper bound: 187.7851767
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7879154
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7740584, upper bound: 187.7740584
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7740584, upper bound: 187.7740584
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7875018
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7851767, upper bound: 187.7851767
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7851767, upper bound: 187.7855504
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5399757
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5348344, upper bound: 187.5414014
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5777814
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5776496
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5283737, upper bound: 187.5283737
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5283737, upper bound: 187.5283737
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5283737, upper bound: 187.5283737
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5310697, upper bound: 187.5283737
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5283737, upper bound: 187.5283737
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5283737, upper bound: 187.5283737
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5345739, upper bound: 187.5343558
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5347453
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5347457
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6014753
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6014753
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5358882, upper bound: 187.5343558
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6015862
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6014758
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6014753
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6014753
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6015995
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6014753
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6014753
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6014753
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6014758
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6014753, upper bound: 187.6014753
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5343558
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5343558, upper bound: 187.5348909
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7694885, upper bound: 187.7694885
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7694885, upper bound: 187.7694885
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=232.61239624023438
rel_dist={3: [-187.90965608592424, 187.9096560859242]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8865291, upper bound: 187.8865291
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8865291, upper bound: 187.8949839
time: 0.79 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.46 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 3, lower bound: -187.8865291, upper bound: 187.8865291
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 3, lower bound: -187.8865291, upper bound: 187.8949839

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8839161, upper bound: 187.8839740
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8922471, upper bound: 187.8839161
time: 0.63 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8858666, upper bound: 187.8939519
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8862633, upper bound: 187.8943683
time: 1.32 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.57 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.57
Output dim: 3, lower bound: -187.8839161, upper bound: 187.8839740
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.57
Output dim: 3, lower bound: -187.8922471, upper bound: 187.8839161
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.57
Output dim: 3, lower bound: -187.8858666, upper bound: 187.8939519
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.57
Output dim: 3, lower bound: -187.8862633, upper bound: 187.8943683

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8837860, upper bound: 187.8837860
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8838704, upper bound: 187.8839740
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8843882, upper bound: 187.8783070
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8869195, upper bound: 187.8782393
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6600933, upper bound: 187.6673395
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6601343, upper bound: 187.6658827
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6417815, upper bound: 187.6490210
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6417930, upper bound: 187.6475439
time: 0.57 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.93 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 3, lower bound: -187.8837860, upper bound: 187.8837860
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 3, lower bound: -187.8838704, upper bound: 187.8839740
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 3, lower bound: -187.8843882, upper bound: 187.8783070
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 3, lower bound: -187.8869195, upper bound: 187.8782393
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 3, lower bound: -187.6600933, upper bound: 187.6673395
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 3, lower bound: -187.6601343, upper bound: 187.6658827
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 3, lower bound: -187.6417815, upper bound: 187.6490210
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 3, lower bound: -187.6417930, upper bound: 187.6475439

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6473242, upper bound: 187.6419244
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6487591, upper bound: 187.6419244
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8333750, upper bound: 187.8306375
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8333750, upper bound: 187.8306375
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8741521, upper bound: 187.8783070
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8843882, upper bound: 187.8704743
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8863519, upper bound: 187.8782110
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8869195, upper bound: 187.8782393
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6598747, upper bound: 187.6673395
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6598579, upper bound: 187.6672640
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6204208, upper bound: 187.6257924
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6204208, upper bound: 187.6255536
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6411399, upper bound: 187.6487938
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6427426, upper bound: 187.6444425
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6414426, upper bound: 187.6474939
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6414426, upper bound: 187.6475439
time: 0.60 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 3, lower bound: -187.6473242, upper bound: 187.6419244
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 3, lower bound: -187.6487591, upper bound: 187.6419244
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 3, lower bound: -187.8333750, upper bound: 187.8306375
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 3, lower bound: -187.8333750, upper bound: 187.8306375
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 3, lower bound: -187.8741521, upper bound: 187.8783070
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 3, lower bound: -187.8843882, upper bound: 187.8704743
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 3, lower bound: -187.8863519, upper bound: 187.8782110
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 3, lower bound: -187.8869195, upper bound: 187.8782393
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 3, lower bound: -187.6598747, upper bound: 187.6673395
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 3, lower bound: -187.6598579, upper bound: 187.6672640
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 3, lower bound: -187.6204208, upper bound: 187.6257924
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 3, lower bound: -187.6204208, upper bound: 187.6255536
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 3, lower bound: -187.6411399, upper bound: 187.6487938
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 3, lower bound: -187.6427426, upper bound: 187.6444425
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 3, lower bound: -187.6414426, upper bound: 187.6474939
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 3, lower bound: -187.6414426, upper bound: 187.6475439

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6415876, upper bound: 187.6419244
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6473242, upper bound: 187.6418578
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5806137, upper bound: 187.5805112
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5806137, upper bound: 187.5806008
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8263953, upper bound: 187.8280048
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8327806, upper bound: 187.8261110
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8263953, upper bound: 187.8280048
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8327806, upper bound: 187.8261110
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6439929, upper bound: 187.6427186
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6439978, upper bound: 187.6426574
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8840164, upper bound: 187.8704743
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8843882, upper bound: 187.8699876
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8733416, upper bound: 187.8782110
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8863519, upper bound: 187.8707304
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8832755, upper bound: 187.8658829
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8863205, upper bound: 187.8714159
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6575280, upper bound: 187.6660041
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6575280, upper bound: 187.6648395
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6597731, upper bound: 187.6672639
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6597718, upper bound: 187.6667426
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169939, upper bound: 187.6250718
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6180828, upper bound: 187.6242575
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6175013, upper bound: 187.6248099
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6188836, upper bound: 187.6241912
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6426810, upper bound: 187.6489381
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6412167, upper bound: 187.6481476
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6207836, upper bound: 187.6241817
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6197110, upper bound: 187.6232745
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6410936, upper bound: 187.6474939
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6426583, upper bound: 187.6441431
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396110, upper bound: 187.6472362
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396312, upper bound: 187.6453550
time: 0.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.6415876, upper bound: 187.6419244
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.6473242, upper bound: 187.6418578
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.5806137, upper bound: 187.5805112
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.5806137, upper bound: 187.5806008
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.8263953, upper bound: 187.8280048
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.8327806, upper bound: 187.8261110
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.8263953, upper bound: 187.8280048
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.8327806, upper bound: 187.8261110
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.6439929, upper bound: 187.6427186
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.6439978, upper bound: 187.6426574
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.8840164, upper bound: 187.8704743
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.8843882, upper bound: 187.8699876
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.8733416, upper bound: 187.8782110
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.8863519, upper bound: 187.8707304
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.8832755, upper bound: 187.8658829
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.8863205, upper bound: 187.8714159
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.6575280, upper bound: 187.6660041
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.6575280, upper bound: 187.6648395
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.6597731, upper bound: 187.6672639
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.6597718, upper bound: 187.6667426
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.6169939, upper bound: 187.6250718
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.6180828, upper bound: 187.6242575
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.6175013, upper bound: 187.6248099
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.6188836, upper bound: 187.6241912
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.6426810, upper bound: 187.6489381
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.6412167, upper bound: 187.6481476
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.6207836, upper bound: 187.6241817
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.6197110, upper bound: 187.6232745
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.6410936, upper bound: 187.6474939
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.6426583, upper bound: 187.6441431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.6396110, upper bound: 187.6472362
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 3, lower bound: -187.6396312, upper bound: 187.6453550

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6392746, upper bound: 187.6400136
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6392746, upper bound: 187.6397906
time: 0.67 seconds

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
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5804595, upper bound: 187.5804595
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5804595, upper bound: 187.5804595
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5346237
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5348825
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5804595, upper bound: 187.5806008
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5804595, upper bound: 187.5804595
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8197283, upper bound: 187.8195987
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8193938, upper bound: 187.8213217
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8321200, upper bound: 187.8254501
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8319947, upper bound: 187.8254501
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8197283, upper bound: 187.8195987
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8195616, upper bound: 187.8213217
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8261868, upper bound: 187.8193938
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8258519, upper bound: 187.8193938
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6410191, upper bound: 187.6397154
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6432832, upper bound: 187.6394677
time: 0.65 seconds

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
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5777814
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5777814
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8057350, upper bound: 187.8057442
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8057350, upper bound: 187.8057442
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6598472, upper bound: 187.6598472
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6636052, upper bound: 187.6598472
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6441020, upper bound: 187.6414963
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6441020, upper bound: 187.6414480
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8832393, upper bound: 187.8682296
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8841082, upper bound: 187.8658829
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6393688, upper bound: 187.6388787
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6466892, upper bound: 187.6388787
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6662641, upper bound: 187.6575671
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6674840, upper bound: 187.6572168
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6570670, upper bound: 187.6660041
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6575280, upper bound: 187.6612420
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6569374, upper bound: 187.6648394
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6569374, upper bound: 187.6646321
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6594421, upper bound: 187.6672639
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6597768, upper bound: 187.6622133
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6596863, upper bound: 187.6667426
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6597731, upper bound: 187.6594421
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6161216, upper bound: 187.6250718
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6166678, upper bound: 187.6219597
time: 0.67 seconds

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
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6177975, upper bound: 187.6242575
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6179227, upper bound: 187.6184464
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169317, upper bound: 187.6248099
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6171681, upper bound: 187.6216158
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6188836, upper bound: 187.6241912
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6188583, upper bound: 187.6192526
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6384168, upper bound: 187.6474370
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6392747, upper bound: 187.6462543
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6410383, upper bound: 187.6410383
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6410383, upper bound: 187.6410383
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6204110, upper bound: 187.6241231
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6207301, upper bound: 187.6233635
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6203280, upper bound: 187.6232273
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6200985, upper bound: 187.6195732
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6410936, upper bound: 187.6474939
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6410936, upper bound: 187.6471172
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6391385, upper bound: 187.6438743
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6389898, upper bound: 187.6386666
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6392799, upper bound: 187.6471769
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6392799, upper bound: 187.6421878
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396312, upper bound: 187.6453550
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396010, upper bound: 187.6386666
time: 0.66 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6392746, upper bound: 187.6400136
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6392746, upper bound: 187.6397906
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.5804595, upper bound: 187.5804595
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.5804595, upper bound: 187.5804595
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5346237
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5348825
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.5804595, upper bound: 187.5806008
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.5804595, upper bound: 187.5804595
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.8197283, upper bound: 187.8195987
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.8193938, upper bound: 187.8213217
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.8321200, upper bound: 187.8254501
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.8319947, upper bound: 187.8254501
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.8197283, upper bound: 187.8195987
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.8195616, upper bound: 187.8213217
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.8261868, upper bound: 187.8193938
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.8258519, upper bound: 187.8193938
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6410191, upper bound: 187.6397154
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6432832, upper bound: 187.6394677
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5777814
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5777814
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.8057350, upper bound: 187.8057442
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.8057350, upper bound: 187.8057442
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6598472, upper bound: 187.6598472
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6636052, upper bound: 187.6598472
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6441020, upper bound: 187.6414963
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6441020, upper bound: 187.6414480
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.8832393, upper bound: 187.8682296
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.8841082, upper bound: 187.8658829
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6393688, upper bound: 187.6388787
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6466892, upper bound: 187.6388787
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6662641, upper bound: 187.6575671
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6674840, upper bound: 187.6572168
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6570670, upper bound: 187.6660041
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6575280, upper bound: 187.6612420
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6569374, upper bound: 187.6648394
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6569374, upper bound: 187.6646321
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6594421, upper bound: 187.6672639
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6597768, upper bound: 187.6622133
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6596863, upper bound: 187.6667426
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6597731, upper bound: 187.6594421
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6161216, upper bound: 187.6250718
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6166678, upper bound: 187.6219597
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6177975, upper bound: 187.6242575
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6179227, upper bound: 187.6184464
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6169317, upper bound: 187.6248099
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6171681, upper bound: 187.6216158
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6188836, upper bound: 187.6241912
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6188583, upper bound: 187.6192526
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6384168, upper bound: 187.6474370
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6392747, upper bound: 187.6462543
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6410383, upper bound: 187.6410383
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6410383, upper bound: 187.6410383
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6204110, upper bound: 187.6241231
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6207301, upper bound: 187.6233635
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6203280, upper bound: 187.6232273
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6200985, upper bound: 187.6195732
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6410936, upper bound: 187.6474939
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6410936, upper bound: 187.6471172
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6391385, upper bound: 187.6438743
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6389898, upper bound: 187.6386666
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6392799, upper bound: 187.6471769
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6392799, upper bound: 187.6421878
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6396312, upper bound: 187.6453550
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 3, lower bound: -187.6396010, upper bound: 187.6386666

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6388787, upper bound: 187.6397350
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6388787, upper bound: 187.6397350
time: 0.61 seconds

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
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5804595, upper bound: 187.5805112
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5804595, upper bound: 187.5805941
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5346237
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5346237
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5346973
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5346237, upper bound: 187.5348825
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8189003, upper bound: 187.8187175
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8189314, upper bound: 187.8189223
time: 0.64 seconds

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7716275, upper bound: 187.7716275
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7716275, upper bound: 187.7716275
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8253830, upper bound: 187.8187175
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8248520, upper bound: 187.8187175
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8312013, upper bound: 187.8250812
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8254109, upper bound: 187.8250812
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8194193, upper bound: 187.8191873
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8194535, upper bound: 187.8191187
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8191187, upper bound: 187.8208620
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8191187, upper bound: 187.8191187
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8254352, upper bound: 187.8191187
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8195035, upper bound: 187.8191187
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
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8248520, upper bound: 187.8187175
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8250610, upper bound: 187.8187175
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165274, upper bound: 187.6180492
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165274, upper bound: 187.6165274
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6388787, upper bound: 187.6394677
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6432832, upper bound: 187.6394572
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5283737, upper bound: 187.5286500
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5283737, upper bound: 187.5286499
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5777814
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8025019, upper bound: 187.8036200
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8025019, upper bound: 187.8025019
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6593749, upper bound: 187.6593749
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6593749, upper bound: 187.6593749
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6628853, upper bound: 187.6593749
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6594161, upper bound: 187.6593749
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6409754, upper bound: 187.6409843
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6436217, upper bound: 187.6409808
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5775126, upper bound: 187.5775126
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6576661, upper bound: 187.6582347
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6652788, upper bound: 187.6582335
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6572168, upper bound: 187.6572168
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6660928, upper bound: 187.6572168
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6244853, upper bound: 187.6165274
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6245048, upper bound: 187.6165274
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.88 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=232.61239624023438
rel_dist={3: [-187.89872093335524, 187.8987209333552]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1137.91 seconds
