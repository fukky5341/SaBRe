## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_5.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 554.967677004936


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141)
1: (-208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797)
2: (-176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039)
3: (-185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579)
4: (-158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706)

## BASE Result
execution time: IAR + LP analysis = 2.09 + 2.21 = 4.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -554.9912953, upper bound: 554.9912953


# Binary Search by BASE starts (time budget: 1195.71 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=619.6494140625
rel_dist={0: [-554.9911089992902, 554.9911089992902]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=619.6494140625
rel_dist={0: [-554.9907236424904, 554.9907236424906]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=619.6494140625
rel_dist={0: [-554.990153595014, 554.9901535950139]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=619.6494140625
rel_dist={0: [-554.988966335711, 554.988966335711]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=619.6494140625
rel_dist={0: [-554.9880975414051, 554.9880975414051]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=619.6494140625
rel_dist={0: [-554.9876280275749, 554.9876280275748]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=619.6494140625
rel_dist={0: [-554.9873672980016, 554.9873672980016]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=619.6494140625
rel_dist={0: [-554.9872352391235, 554.9872352391233]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=619.6494140625
rel_dist={0: [-554.9871688329926, 554.9871688329922]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=619.6494140625
rel_dist={0: [-554.9871355495204, 554.9871355495204]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=619.6494140625
rel_dist={0: [-554.9871181509133, 554.9871181509134]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=619.6494140625
rel_dist={0: [-554.9871092291382, 554.9871092291382]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=619.6494140625
rel_dist={0: [-554.9871047682793, 554.9871047682793]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=619.6494140625
rel_dist={0: [-554.9871025379068, 554.9871025379066]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=619.6494140625
rel_dist={0: [-554.9871014240496, 554.9871014228313]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=619.6494140625
rel_dist={0: [-554.9871008661073, 554.9871008655066]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=619.6494140625
rel_dist={0: [-554.9871005875302, 554.9871006005683]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=619.6494140625
rel_dist={0: [-554.9871004489975, 554.9871004644856]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=619.6494140625
rel_dist={0: [-554.9871004039769, 554.9871004148458]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=619.6494140625
rel_dist={0: [-554.987100444383, 554.9871004366394]}

## Binary Search Result
Binary search time: 89.26 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1106.44 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9794534
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9794534
time: 1.06 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.29 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.29
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9794534
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.29
Output dim: 0, lower bound: -554.9794534, upper bound: 554.9794534

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9723813
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723813, upper bound: 554.9718254
time: 1.14 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9723813
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723813, upper bound: 554.9718254
time: 1.14 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.26 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.26
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9723813
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.26
Output dim: 0, lower bound: -554.9723813, upper bound: 554.9718254
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.26
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9723813
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.26
Output dim: 0, lower bound: -554.9723813, upper bound: 554.9718254

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9723813
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9718254
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9718254
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723813, upper bound: 554.9718253
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9723813
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9718253
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9718253
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723813, upper bound: 554.9718253
time: 1.25 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.56 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9723813
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9718254
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9718254
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -554.9723813, upper bound: 554.9718253
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9723813
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9718253
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9718253
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -554.9723813, upper bound: 554.9718253

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689829
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689829
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689829, upper bound: 554.9689722
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
time: 1.00 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689829
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689829
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -554.9689829, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.93
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9693839
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689829
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689829, upper bound: 554.9689722
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9693839, upper bound: 554.9689722
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9693839
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689829
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689829, upper bound: 554.9689722
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9693839, upper bound: 554.9689722
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
time: 1.02 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9693839
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689829
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689829, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9693839, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9693839
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689829
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689829, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9693839, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9693839
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689986
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689829
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689829, upper bound: 554.9689722
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689986, upper bound: 554.9689722
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9693839, upper bound: 554.9689722
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689899, upper bound: 554.9689722
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689748, upper bound: 554.9689722
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689748
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689899
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9693839
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689986
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689829
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689829, upper bound: 554.9689722
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689986, upper bound: 554.9689722
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9693839, upper bound: 554.9689722
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
time: 1.38 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9693839
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689986
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689829
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689829, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689986, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9693839, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689899, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689748, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689748
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689899
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9693839
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689986
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689829
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689829, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689986, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9693839, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.87
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.14 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 5.49 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.49
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689748, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689748
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699735
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689899
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9693839
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689986
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689829
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689829, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689986, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9693839, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.49
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9689722
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=619.6494140625
rel_dist={0: [-554.9911089992902, 554.9911089992902]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794499, upper bound: 554.9794499
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794499, upper bound: 554.9794499
time: 0.84 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.86 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.86
Output dim: 0, lower bound: -554.9794499, upper bound: 554.9794499
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.86
Output dim: 0, lower bound: -554.9794499, upper bound: 554.9794499

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9723812
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9718253
time: 1.50 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9723812
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723812, upper bound: 554.9718253
time: 1.04 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.98 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.98
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9723812
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.98
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9718253
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.98
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9723812
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.98
Output dim: 0, lower bound: -554.9723812, upper bound: 554.9718253

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9723812
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9718254
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9718253
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723812, upper bound: 554.9718253
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9723812
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9718254
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9718253
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723812, upper bound: 554.9718253
time: 1.18 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.56 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9723812
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9718254
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9718253
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -554.9723812, upper bound: 554.9718253
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9723812
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9718254
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -554.9718254, upper bound: 554.9718253
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 0, lower bound: -554.9723812, upper bound: 554.9718253

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689812
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689812, upper bound: 554.9689722
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699662, upper bound: 554.9689722
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699662, upper bound: 554.9689722
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689812
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689812, upper bound: 554.9689722
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699662, upper bound: 554.9689722
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699662, upper bound: 554.9689722
time: 1.14 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689812
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -554.9689812, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -554.9699662, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -554.9699662, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689812
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -554.9689812, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -554.9699662, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.18
Output dim: 0, lower bound: -554.9699662, upper bound: 554.9689722

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9693777
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689812
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689812, upper bound: 554.9689722
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699662, upper bound: 554.9689722
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9693777
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689812
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689812, upper bound: 554.9689722
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.21 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9693777
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689812
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689812, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9699662, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9693777
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689812
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689812, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9693777
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689969
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689812
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689812, upper bound: 554.9689722
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689881, upper bound: 554.9689722
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689748, upper bound: 554.9689722
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689748
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689881
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9693777
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689969
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689812
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
time: 1.42 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9693777
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689969
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689812
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689812, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689881, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689748, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689748
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689881
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9693777
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689969
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689812
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
time: 1.34 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 5.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 0, lower bound: -554.9009069, upper bound: 554.9009069
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689881, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689748, upper bound: 554.9689722
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689748
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9699662
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689881
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9693777
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689969
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689812
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 0, lower bound: -554.9689722, upper bound: 554.9689722
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=619.6494140625
rel_dist={0: [-554.9907236424904, 554.9907236424906]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9788198, upper bound: 554.9788198
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9788198, upper bound: 554.9788198
time: 1.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.26 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.26
Output dim: 0, lower bound: -554.9788198, upper bound: 554.9788198
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.26
Output dim: 0, lower bound: -554.9788198, upper bound: 554.9788198

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9719431
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9711433
time: 1.34 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9719431
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9711433
time: 1.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.38 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.38
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9719431
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.38
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9711433
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.38
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9719431
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.38
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9711433

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9719431
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9711433
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9711433
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9711433
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9719431
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9711433
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9711433
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9711433
time: 1.10 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.47 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9719431
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9711433
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9711433
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9711433
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9719431
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9711433
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9711433
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 0, lower bound: -554.9711433, upper bound: 554.9711433

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685657
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685657, upper bound: 554.9685494
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9695519, upper bound: 554.9685494
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9695519, upper bound: 554.9685494
time: 1.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685657
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685657, upper bound: 554.9685494
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9695519, upper bound: 554.9685494
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9695519, upper bound: 554.9685494
time: 1.61 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.11
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.11
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.11
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.11
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685657
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.11
Output dim: 0, lower bound: -554.9685657, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.11
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.11
Output dim: 0, lower bound: -554.9695519, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.11
Output dim: 0, lower bound: -554.9695519, upper bound: 554.9685494
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.11
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.11
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.11
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.11
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685657
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.11
Output dim: 0, lower bound: -554.9685657, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.11
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.11
Output dim: 0, lower bound: -554.9695519, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.11
Output dim: 0, lower bound: -554.9695519, upper bound: 554.9685494

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9689235
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685657
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689235, upper bound: 554.9685494
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9695519, upper bound: 554.9685494
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9695519, upper bound: 554.9685494
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9689235
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685657
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685657, upper bound: 554.9685494
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689235, upper bound: 554.9685494
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9695519, upper bound: 554.9685494
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9695519, upper bound: 554.9685494
time: 1.18 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9689235
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685657
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9689235, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9695519, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9695519, upper bound: 554.9685494
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9689235
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685657
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685657, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9689235, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9695519, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 0, lower bound: -554.9695519, upper bound: 554.9685494

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9689235
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685798
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685657
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685657, upper bound: 554.9685494
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689235, upper bound: 554.9685494
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685689, upper bound: 554.9685494
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9695519, upper bound: 554.9685494
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685614
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685689
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9689235
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685798
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685657
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141
1: -208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797
2: -176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039
3: -185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579
4: -158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
time: 1.15 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9689235
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685798
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685657
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685657, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9689235, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685689, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9695519, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685614
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9695519
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685689
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9689235
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685798
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685657
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -554.9689235, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -554.9695519, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -554.9685494, upper bound: 554.9685494
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -554.9695519, upper bound: 554.9685494
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=619.6494140625
rel_dist={0: [-554.990153595014, 554.9901535950139]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1106.47 seconds
