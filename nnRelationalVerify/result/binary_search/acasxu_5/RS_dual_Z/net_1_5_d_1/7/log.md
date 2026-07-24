## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_5.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 339.77104719722996


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423)
1: (-124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621)
2: (-105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148)
3: (-110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960)
4: (-94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043)

## BASE Result
execution time: IAR + LP analysis = 2.29 + 2.32 = 4.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -339.8056876, upper bound: 339.8056876


# Binary Search by BASE starts (time budget: 1195.39 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=385.80084228515625
rel_dist={0: [-339.80568762378186, 339.805687623782]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=385.80084228515625
rel_dist={0: [-339.8055350744037, 339.8055350744037]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=385.80084228515625
rel_dist={0: [-339.8051238459851, 339.80512384598524]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=385.80084228515625
rel_dist={0: [-339.8046744597558, 339.8046744597558]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=385.80084228515625
rel_dist={0: [-339.80427711404343, 339.80427711404354]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=385.80084228515625
rel_dist={0: [-339.80406740868005, 339.8040674086801]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=385.80084228515625
rel_dist={0: [-339.8039615610313, 339.8039615610313]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=385.80084228515625
rel_dist={0: [-339.8039086372072, 339.8039086372073]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=385.80084228515625
rel_dist={0: [-339.80388211805257, 339.80388211805234]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=385.80084228515625
rel_dist={0: [-339.8038687415637, 339.80386874156375]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=385.80084228515625
rel_dist={0: [-339.80386204377714, 339.80386204377714]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=385.80084228515625
rel_dist={0: [-339.8038586491076, 339.8038586491076]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=385.80084228515625
rel_dist={0: [-339.80385694234457, 339.8038569423445]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=385.80084228515625
rel_dist={0: [-339.80385608897905, 339.80385608897905]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=385.80084228515625
rel_dist={0: [-339.8038556623275, 339.8038556623276]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=385.80084228515625
rel_dist={0: [-339.80385544906244, 339.80385544906244]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=385.80084228515625
rel_dist={0: [-339.8038553425432, 339.8038553425432]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=385.80084228515625
rel_dist={0: [-339.8038552977798, 339.8038552943109]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=385.80084228515625
rel_dist={0: [-339.8038553014052, 339.80385528113925]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=385.80084228515625
rel_dist={0: [-339.80385530851163, 339.8038553170004]}

## Binary Search Result
Binary search time: 89.14 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1106.25 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.46 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.07 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.07
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.07
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7674305, upper bound: 339.7674243
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7674305, upper bound: 339.7674243
time: 1.06 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7674243, upper bound: 339.7674305
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7674243, upper bound: 339.7674305
time: 1.17 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.70 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 4.70
Output dim: 0, lower bound: -339.7674305, upper bound: 339.7674243
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 4.70
Output dim: 0, lower bound: -339.7674305, upper bound: 339.7674243
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 4.70
Output dim: 0, lower bound: -339.7674243, upper bound: 339.7674305
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 4.70
Output dim: 0, lower bound: -339.7674243, upper bound: 339.7674305
Binary search (step 0): status=Status.VERIFIED, low=0.5000000, high=1.0000000, mid=0.5000000, abs_max=385.80084228515625
rel_dist={0: [-339.80568762378186, 339.805687623782]}

## Binary search (step 1) starts
Candidate diff: 0.7500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
time: 1.09 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.68
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.68
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.86 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.82 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.05 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.05
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.05
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.05
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.05
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.89 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.13 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 2.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 2.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
time: 1.06 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.54 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.54
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.54
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.54
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.54
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.54
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.54
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.54
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.54
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.54
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.54
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.54
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.54
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.54
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.54
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.54
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.54
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
Binary search (step 1): status=Status.VERIFIED, low=0.7500000, high=1.0000000, mid=0.7500000, abs_max=385.80084228515625
rel_dist={0: [-339.80568762378186, 339.805687623782]}

## Binary search (step 2) starts
Candidate diff: 0.8750000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 0.96 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.14 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.14
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.14
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.89 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.90 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.14 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.14
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.14
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.14
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.14
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.91 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.04 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 1.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
time: 1.02 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.72 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.72
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.72
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.72
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.72
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.72
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.72
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.72
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.72
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.72
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.72
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.72
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.72
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.72
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.72
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.72
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.72
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
Binary search (step 2): status=Status.VERIFIED, low=0.8750000, high=1.0000000, mid=0.8750000, abs_max=385.80084228515625
rel_dist={0: [-339.80568762378186, 339.805687623782]}

## Binary search (step 3) starts
Candidate diff: 0.9375000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.00 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.29 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.29
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.29
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.93 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.93 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.12
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.12
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.12
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.12
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.90 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.02 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
time: 1.06 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.74
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.74
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.74
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.74
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.74
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.74
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.74
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.74
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.74
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.74
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.74
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.74
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.74
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.74
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.74
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.74
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
Binary search (step 3): status=Status.VERIFIED, low=0.9375000, high=1.0000000, mid=0.9375000, abs_max=385.80084228515625
rel_dist={0: [-339.80568762378186, 339.80568762378186]}

## Binary search (step 4) starts
Candidate diff: 0.9687500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
time: 1.12 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.43 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.43
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.43
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.75 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.76 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.80 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.80
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.80
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.80
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.80
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.79 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.24 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
time: 1.07 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.62
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.62
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.62
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.62
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.62
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.62
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.62
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.62
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.62
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.62
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.62
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.62
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.62
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.62
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.62
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.62
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
Binary search (step 4): status=Status.VERIFIED, low=0.9687500, high=1.0000000, mid=0.9687500, abs_max=385.80084228515625
rel_dist={0: [-339.80568762378186, 339.805687623782]}

## Binary search (step 5) starts
Candidate diff: 0.9843750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
time: 1.00 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.22 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.22
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.22
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.96 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.96 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.14 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.14
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.14
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.14
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.14
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.90 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.19 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
time: 1.12 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
Binary search (step 5): status=Status.VERIFIED, low=0.9843750, high=1.0000000, mid=0.9843750, abs_max=385.80084228515625
rel_dist={0: [-339.80568762378186, 339.805687623782]}

## Binary search (step 6) starts
Candidate diff: 0.9921875


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.21 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.62 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.62
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.62
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 1.14 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 1.15 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.73 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.73
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.73
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.73
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.73
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.82 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.18 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.18
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.18
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.18
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.18
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.18
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.18
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.18
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.18
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
time: 1.05 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.41
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.41
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.41
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.41
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.41
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.41
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.41
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.41
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.41
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.41
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.41
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.41
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.41
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.41
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.41
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.41
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
Binary search (step 6): status=Status.VERIFIED, low=0.9921875, high=1.0000000, mid=0.9921875, abs_max=385.80084228515625
rel_dist={0: [-339.80568762378186, 339.805687623782]}

## Binary search (step 7) starts
Candidate diff: 0.9960938


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
time: 1.13 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.53 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.53
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.53
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.93 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.93 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.32 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.32
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.32
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.32
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.32
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.94 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.34 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
time: 0.93 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.55
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.55
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.55
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.55
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.55
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.55
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.55
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.55
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.55
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.55
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.55
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.55
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.55
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.55
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.55
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.55
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
Binary search (step 7): status=Status.VERIFIED, low=0.9960938, high=1.0000000, mid=0.9960938, abs_max=385.80084228515625
rel_dist={0: [-339.80568762378186, 339.80568762378186]}

## Binary search (step 8) starts
Candidate diff: 0.9980469


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.37 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.94 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.94
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.94
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 1.07 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 1.07 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.48 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.48
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.48
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.48
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.48
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.89 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.02 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
time: 1.06 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.58
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.58
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.58
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.58
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.58
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.58
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.58
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.58
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.58
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.58
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.58
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.58
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.58
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.58
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.58
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.58
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
Binary search (step 8): status=Status.VERIFIED, low=0.9980469, high=1.0000000, mid=0.9980469, abs_max=385.80084228515625
rel_dist={0: [-339.80568762378186, 339.80568762378186]}

## Binary search (step 9) starts
Candidate diff: 0.9990234


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.10 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.44 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.44
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.44
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 1.06 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 1.05 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.42 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.42
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.42
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.42
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.42
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.93 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.10 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
time: 0.98 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.43
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.43
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.43
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.43
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.43
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.43
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.43
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.43
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.43
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.43
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.43
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.43
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.43
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.43
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.43
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.43
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
Binary search (step 9): status=Status.VERIFIED, low=0.9990234, high=1.0000000, mid=0.9990234, abs_max=385.80084228515625
rel_dist={0: [-339.80568762378186, 339.80568762378186]}

## Binary search (step 10) starts
Candidate diff: 0.9995117


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.18 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.63 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.63
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.63
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.87 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.89 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.99 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.99
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.99
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.99
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.99
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.96 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.13 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
time: 0.91 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
Binary search (step 10): status=Status.VERIFIED, low=0.9995117, high=1.0000000, mid=0.9995117, abs_max=385.80084228515625
rel_dist={0: [-339.80568762378186, 339.80568762378186]}

## Binary search (step 11) starts
Candidate diff: 0.9997559


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
time: 1.10 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.30 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.30
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.30
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.96 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.96 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.20 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.20
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.20
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.20
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.20
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.82 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.09 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
time: 1.15 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.64
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.64
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.64
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.64
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.64
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.64
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.64
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.64
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.64
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.64
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.64
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.64
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.64
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.64
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.64
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.64
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
Binary search (step 11): status=Status.VERIFIED, low=0.9997559, high=1.0000000, mid=0.9997559, abs_max=385.80084228515625
rel_dist={0: [-339.80568762378186, 339.80568762378186]}

## Binary search (step 12) starts
Candidate diff: 0.9998779


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.05 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.28 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.28
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.28
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.88 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.87 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.08 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.92 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.31 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.31
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.31
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.31
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.31
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.31
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.31
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.31
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.31
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
time: 1.01 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.34
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.34
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.34
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.34
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.34
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.34
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.34
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.34
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.34
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.34
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.34
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.34
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.34
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.34
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.34
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.34
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
Binary search (step 12): status=Status.VERIFIED, low=0.9998779, high=1.0000000, mid=0.9998779, abs_max=385.80084228515625
rel_dist={0: [-339.80568762378186, 339.805687623782]}

## Binary search (step 13) starts
Candidate diff: 0.9999390


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
time: 1.12 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.53 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.53
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.53
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.89 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.88 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.19 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.19
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.19
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.19
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.19
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.93 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.34 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
time: 1.04 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.65
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.65
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.65
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.65
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.65
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.65
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.65
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.65
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.65
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.65
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.65
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.65
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.65
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.65
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.65
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.65
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
Binary search (step 13): status=Status.VERIFIED, low=0.9999390, high=1.0000000, mid=0.9999390, abs_max=385.80084228515625
rel_dist={0: [-339.80568762378186, 339.805687623782]}

## Binary search (step 14) starts
Candidate diff: 0.9999695


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
time: 1.09 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.29 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.29
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.29
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 1.01 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 1.01 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.48 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.48
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.48
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.48
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.48
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.88 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.20 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
time: 0.99 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.50
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.50
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.50
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.50
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.50
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.50
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.50
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.50
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.50
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.50
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.50
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.50
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.50
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.50
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.50
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.50
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
Binary search (step 14): status=Status.VERIFIED, low=0.9999695, high=1.0000000, mid=0.9999695, abs_max=385.80084228515625
rel_dist={0: [-339.80568762378186, 339.80568762378186]}

## Binary search (step 15) starts
Candidate diff: 0.9999847


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
time: 1.38 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.97 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.97
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.97
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.78 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.77 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.00 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.00
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.00
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.00
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.00
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.88 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.07 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
time: 1.24 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.95
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.95
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.95
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.95
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.95
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.95
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.95
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.95
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.95
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.95
Output dim: 0, lower bound: -339.7636017, upper bound: 339.7636069
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.95
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.95
Output dim: 0, lower bound: -339.7636039, upper bound: 339.7635426
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.95
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.95
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636067
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.95
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.95
Output dim: 0, lower bound: -339.7635927, upper bound: 339.7635981
Binary search (step 15): status=Status.VERIFIED, low=0.9999847, high=1.0000000, mid=0.9999847, abs_max=385.80084228515625
rel_dist={0: [-339.80568762378186, 339.80568762378186]}

## Binary search (step 16) starts
Candidate diff: 0.9999924


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.03 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.27 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.27
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.27
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.87 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
time: 0.86 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.07 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.07
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.07
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.07
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.07
Output dim: 0, lower bound: -339.7948801, upper bound: 339.7948801

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
time: 0.87 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.11 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.11
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.11
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.11
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.11
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.11
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.11
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.11
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.11
Output dim: 0, lower bound: -339.7947080, upper bound: 339.7947080

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635981, upper bound: 339.7635927
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635920, upper bound: 339.7635927
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636067, upper bound: 339.7635426
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7635426, upper bound: 339.7636039
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7636069, upper bound: 339.7636017
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -88.6961975, 297.1046753, -88.6961975, 297.1046753, -385.8008423, 385.8008423
1: -124.4471970, 294.8176575, -124.4471970, 294.8176575, -419.2648621, 419.2648621
2: -105.5478058, 324.6724243, -105.5478058, 324.6724243, -430.2202148, 430.2202148
3: -110.7164154, 421.9519958, -110.7164154, 421.9519958, -532.6683960, 532.6683960
4: -94.5076294, 383.5692749, -94.5076294, 383.5692749, -478.0769043, 478.0769043

Time for backsubstitution: 2.26 seconds
Binary search (step 16): status=Status.UNKNOWN, low=0.9999847, high=0.9999924, mid=0.9999924, abs_max=385.80084228515625
rel_dist={0: [-339.80568762378186, 339.80568762378186]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.9999847412109375
execution time: 1108.36 seconds
