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
execution time: IAR + LP analysis = 2.45 + 2.43 = 4.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -339.8056876, upper bound: 339.8056876


# Binary Search by BASE starts (time budget: 1195.12 seconds, max iter: 100)

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
Binary search time: 93.41 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1101.70 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042672, upper bound: 339.8056352
time: 1.27 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.17 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.64 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.64
Output dim: 0, lower bound: -339.8042672, upper bound: 339.8056352
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.64
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -88.6961975, 297.1046753, -382.1996155, 372.8666687
1: -119.3920059, 282.0884705, -124.4471970, 294.8176575, -414.2096252, 406.5356750
2: -101.2751236, 310.7092590, -105.5478058, 324.6724243, -425.9474792, 416.2570801
3: -106.2089005, 403.8296509, -110.7164154, 421.9519958, -528.1608887, 514.5460205
4: -90.6926956, 367.2429504, -94.5076294, 383.5692749, -474.2619629, 461.7505798

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042183, upper bound: 339.8042183
time: 1.23 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
time: 1.05 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -85.5067825, 287.2420959, -88.6961975, 297.1046753, -382.6114502, 375.9382324
1: -120.1114273, 284.9082947, -124.4471970, 294.8176575, -414.9290466, 409.3554993
2: -101.8361511, 313.7402954, -105.5478058, 324.6724243, -426.5085449, 419.2880859
3: -106.8467102, 407.9730225, -110.7164154, 421.9519958, -528.7987061, 518.6894531
4: -91.2148666, 370.7578735, -94.5076294, 383.5692749, -474.7841492, 465.2655029

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.10 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.24 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.91 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.91
Output dim: 0, lower bound: -339.8042183, upper bound: 339.8042183
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.91
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.91
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.91
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -85.0949707, 284.1704712, -369.2654419, 369.2654114
1: -119.3920059, 282.0884705, -119.3920059, 282.0884705, -401.4804077, 401.4804077
2: -101.2751236, 310.7092590, -101.2751236, 310.7092590, -411.9843445, 411.9843445
3: -106.2089005, 403.8296509, -106.2089005, 403.8296509, -510.0385437, 510.0385437
4: -90.6926956, 367.2429504, -90.6926956, 367.2429504, -457.9356384, 457.9356384

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7708530, upper bound: 339.7976800
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8024352, upper bound: 339.8034518
time: 1.18 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -85.5067825, 287.2420959, -372.3370361, 369.6772461
1: -119.3920059, 282.0884705, -120.1114273, 284.9082947, -404.3002319, 402.1998596
2: -101.2751236, 310.7092590, -101.8361511, 313.7402954, -415.0153809, 412.5454102
3: -106.2089005, 403.8296509, -106.8467102, 407.9730225, -514.1819458, 510.6763611
4: -90.6926956, 367.2429504, -91.2148666, 370.7578735, -461.4505615, 458.4578247

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7708530, upper bound: 339.7976800
time: 1.48 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8024352, upper bound: 339.8034518
time: 1.16 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -85.5067825, 287.2420959, -85.0949707, 284.1704712, -369.6772461, 372.3370361
1: -120.1114273, 284.9082947, -119.3920059, 282.0884705, -402.1998596, 404.3002319
2: -101.8361511, 313.7402954, -101.2751236, 310.7092590, -412.5454102, 415.0153809
3: -106.8467102, 407.9730225, -106.2089005, 403.8296509, -510.6763611, 514.1819458
4: -91.2148666, 370.7578735, -90.6926956, 367.2429504, -458.4578247, 461.4505310

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7915542, upper bound: 339.7986696
time: 1.26 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8011052, upper bound: 339.8011052
time: 1.21 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -85.5067825, 287.2420959, -85.5067825, 287.2420959, -372.7488708, 372.7488708
1: -120.1114273, 284.9082947, -120.1114273, 284.9082947, -405.0196838, 405.0196838
2: -101.8361511, 313.7402954, -101.8361511, 313.7402954, -415.5764465, 415.5764465
3: -106.8467102, 407.9730225, -106.8467102, 407.9730225, -514.8197021, 514.8197021
4: -91.2148666, 370.7578735, -91.2148666, 370.7578735, -461.9727173, 461.9727173

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7915542, upper bound: 339.7986696
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8011052, upper bound: 339.8011052
time: 0.92 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.61 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.61
Output dim: 0, lower bound: -339.7708530, upper bound: 339.7976800
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.61
Output dim: 0, lower bound: -339.8024352, upper bound: 339.8034518
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.61
Output dim: 0, lower bound: -339.7708530, upper bound: 339.7976800
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.61
Output dim: 0, lower bound: -339.8024352, upper bound: 339.8034518
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.61
Output dim: 0, lower bound: -339.7915542, upper bound: 339.7986696
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.61
Output dim: 0, lower bound: -339.8011052, upper bound: 339.8011052
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.61
Output dim: 0, lower bound: -339.7915542, upper bound: 339.7986696
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.61
Output dim: 0, lower bound: -339.8011052, upper bound: 339.8011052

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -76.5251465, 258.0331726, -85.0949707, 284.1704712, -360.6956177, 343.1280823
1: -107.2658615, 255.8545990, -119.3920059, 282.0884705, -389.3543091, 375.2466125
2: -90.9418640, 281.8870544, -101.2751236, 310.7092590, -401.6511230, 383.1621704
3: -95.5396729, 366.7406006, -106.2089005, 403.8296509, -499.3692932, 472.9494934
4: -81.6038895, 333.3200378, -90.6926956, 367.2429504, -448.8468323, 424.0127258

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7674618, upper bound: 339.7674618
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7674618, upper bound: 339.7990138
time: 1.33 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -84.2214584, 281.0291443, -85.0949707, 284.1704712, -368.3919373, 366.1240845
1: -118.1812286, 278.9996948, -119.3920059, 282.0884705, -400.2696838, 398.3916626
2: -100.2603912, 307.3133545, -101.2751236, 310.7092590, -410.9696045, 408.5884399
3: -105.1281967, 399.3886414, -106.2089005, 403.8296509, -508.9578247, 505.5975342
4: -89.7864609, 363.2832031, -90.6926956, 367.2429504, -457.0293884, 453.9758911

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7990138, upper bound: 339.7732298
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7990138, upper bound: 339.8047818
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -76.5251465, 258.0331726, -85.5067825, 287.2420959, -363.7672424, 343.5399475
1: -107.2658615, 255.8545990, -120.1114273, 284.9082947, -392.1741333, 375.9660339
2: -90.9418640, 281.8870544, -101.8361511, 313.7402954, -404.6821594, 383.7232056
3: -95.5396729, 366.7406006, -106.8467102, 407.9730225, -503.5126953, 473.5873108
4: -81.6038895, 333.3200378, -91.2148666, 370.7578735, -452.3617554, 424.5349121

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7684476, upper bound: 339.7881328
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7684476, upper bound: 339.7976800
time: 1.24 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -84.2214584, 281.0291443, -85.5067825, 287.2420959, -371.4635620, 366.5359192
1: -118.1812286, 278.9996948, -120.1114273, 284.9082947, -403.0895081, 399.1110535
2: -100.2603912, 307.3133545, -101.8361511, 313.7402954, -414.0006409, 409.1495056
3: -105.1281967, 399.3886414, -106.8467102, 407.9730225, -513.1011963, 506.2353516
4: -89.7864609, 363.2832031, -91.2148666, 370.7578735, -460.5443115, 454.4980774

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7999996, upper bound: 339.7939008
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7999996, upper bound: 339.8034518
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -77.7812729, 263.4603271, -85.0949707, 284.1704712, -361.9517212, 348.5552063
1: -109.1529846, 261.0677795, -119.3920059, 282.0884705, -391.2414246, 380.4597473
2: -92.4850464, 287.5709229, -101.2751236, 310.7092590, -403.1942749, 388.8460083
3: -97.2033539, 374.3854675, -106.2089005, 403.8296509, -501.0330200, 480.5943604
4: -82.9996109, 340.1309814, -90.6926956, 367.2429504, -450.2425537, 430.8236694

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7881328, upper bound: 339.7684476
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7881328, upper bound: 339.7999996
time: 1.25 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -84.3763123, 283.3912354, -85.0949707, 284.1704712, -368.5467834, 368.4862061
1: -118.5572128, 281.0944824, -119.3920059, 282.0884705, -400.6456909, 400.4864502
2: -100.5316162, 309.5450439, -101.2751236, 310.7092590, -411.2408752, 410.8201294
3: -105.4614182, 402.5131226, -106.2089005, 403.8296509, -509.2910767, 508.7220154
4: -90.0528412, 365.8195190, -90.6926956, 367.2429504, -457.2957764, 456.5122070

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7976800, upper bound: 339.7708530
time: 1.22 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7976800, upper bound: 339.8024352
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -77.7812729, 263.4603271, -85.5067825, 287.2420959, -365.0233154, 348.9671021
1: -109.1529846, 261.0677795, -120.1114273, 284.9082947, -394.0612793, 381.1791382
2: -92.4850464, 287.5709229, -101.8361511, 313.7402954, -406.2253113, 389.4070740
3: -97.2033539, 374.3854675, -106.8467102, 407.9730225, -505.1763611, 481.2321777
4: -82.9996109, 340.1309814, -91.2148666, 370.7578735, -453.7574768, 431.3458557

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7891186, upper bound: 339.7881910
time: 1.32 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7891186, upper bound: 339.7986696
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -84.3763123, 283.3912354, -85.5067825, 287.2420959, -371.6184082, 368.8980103
1: -118.5572128, 281.0944824, -120.1114273, 284.9082947, -403.4655151, 401.2059021
2: -100.5316162, 309.5450439, -101.8361511, 313.7402954, -414.2719116, 411.3811951
3: -105.4614182, 402.5131226, -106.8467102, 407.9730225, -513.4344482, 509.3598022
4: -90.0528412, 365.8195190, -91.2148666, 370.7578735, -460.8107300, 457.0343933

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7976990, upper bound: 339.7883617
time: 1.43 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7976990, upper bound: 339.8011052
time: 1.47 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.48 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 5.48
Output dim: 0, lower bound: -339.7674618, upper bound: 339.7674618
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 0, lower bound: -339.7674618, upper bound: 339.7990138
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 0, lower bound: -339.7990138, upper bound: 339.7732298
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 0, lower bound: -339.7990138, upper bound: 339.8047818
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 0, lower bound: -339.7684476, upper bound: 339.7881328
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 0, lower bound: -339.7684476, upper bound: 339.7976800
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 0, lower bound: -339.7999996, upper bound: 339.7939008
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 0, lower bound: -339.7999996, upper bound: 339.8034518
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 0, lower bound: -339.7881328, upper bound: 339.7684476
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 0, lower bound: -339.7881328, upper bound: 339.7999996
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 0, lower bound: -339.7976800, upper bound: 339.7708530
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 0, lower bound: -339.7976800, upper bound: 339.8024352
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 0, lower bound: -339.7891186, upper bound: 339.7881910
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 0, lower bound: -339.7891186, upper bound: 339.7986696
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 0, lower bound: -339.7976990, upper bound: 339.7883617
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 0, lower bound: -339.7976990, upper bound: 339.8011052

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -76.5251465, 258.0331726, -84.2214584, 281.0291443, -357.5542908, 342.2546387
1: -107.2658615, 255.8545990, -118.1812286, 278.9996948, -386.2654724, 374.0358276
2: -90.9418640, 281.8870544, -100.2603912, 307.3133545, -398.2552185, 382.1474304
3: -95.5396729, 366.7406006, -105.1281967, 399.3886414, -494.9283142, 471.8688049
4: -81.6038895, 333.3200378, -89.7864609, 363.2832031, -444.8870850, 423.1064758

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7631526, upper bound: 339.7530758
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7557520, upper bound: 339.7519158
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -84.2214584, 281.0291443, -76.5251465, 258.0331726, -342.2546387, 357.5542908
1: -118.1812286, 278.9996948, -107.2658615, 255.8545990, -374.0358276, 386.2654724
2: -100.2603912, 307.3133545, -90.9418640, 281.8870544, -382.1474609, 398.2552185
3: -105.1281967, 399.3886414, -95.5396729, 366.7406006, -471.8688049, 494.9283142
4: -89.7864609, 363.2832031, -81.6038895, 333.3200378, -423.1064758, 444.8870850

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7978075, upper bound: 339.7726356
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7949819, upper bound: 339.7714272
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -84.2214584, 281.0291443, -84.2214584, 281.0291443, -365.2506104, 365.2506104
1: -118.1812286, 278.9996948, -118.1812286, 278.9996948, -397.1808777, 397.1808472
2: -100.2603912, 307.3133545, -100.2603912, 307.3133545, -407.5736694, 407.5736694
3: -105.1281967, 399.3886414, -105.1281967, 399.3886414, -504.5168457, 504.5168457
4: -89.7864609, 363.2832031, -89.7864609, 363.2832031, -453.0696716, 453.0696716

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7978076, upper bound: 339.8003457
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7949819, upper bound: 339.7991372
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -76.5251465, 258.0331726, -77.7812729, 263.4603271, -339.9854736, 335.8143921
1: -107.2658615, 255.8545990, -109.1529846, 261.0677795, -368.3335876, 365.0075684
2: -90.9418640, 281.8870544, -92.4850464, 287.5709229, -378.5127869, 374.3721008
3: -95.5396729, 366.7406006, -97.2033539, 374.3854675, -469.9251099, 463.9439697
4: -81.6038895, 333.3200378, -82.9996109, 340.1309814, -421.7348633, 416.3196411

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7597992, upper bound: 339.7861661
time: 1.28 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7579234, upper bound: 339.7746415
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7537583, upper bound: 339.7738271
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -76.5251465, 258.0331726, -84.3763123, 283.3912354, -359.9163818, 342.4094849
1: -107.2658615, 255.8545990, -118.5572128, 281.0944824, -388.3603210, 374.4118042
2: -90.9418640, 281.8870544, -100.5316162, 309.5450439, -400.4869080, 382.4186707
3: -95.5396729, 366.7406006, -105.4614182, 402.5131226, -498.0527649, 472.2020264
4: -81.6038895, 333.3200378, -90.0528412, 365.8195190, -447.4234009, 423.3728638

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7597992, upper bound: 339.7960720
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7579234, upper bound: 339.7840502
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7537583, upper bound: 339.7832358
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -84.2214584, 281.0291443, -77.7812729, 263.4603271, -347.6817627, 358.8104248
1: -118.1812286, 278.9996948, -109.1529846, 261.0677795, -379.2489624, 388.1526489
2: -100.2603912, 307.3133545, -92.4850464, 287.5709229, -387.8312683, 399.7983704
3: -105.1281967, 399.3886414, -97.2033539, 374.3854675, -479.5136414, 496.5919800
4: -89.7864609, 363.2832031, -82.9996109, 340.1309814, -429.9174500, 446.2828064

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7985539, upper bound: 339.7908447
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7957283, upper bound: 339.7896363
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -84.2214584, 281.0291443, -84.3763123, 283.3912354, -367.6127014, 365.4054565
1: -118.1812286, 278.9996948, -118.5572128, 281.0944824, -399.2756653, 397.5569153
2: -100.2603912, 307.3133545, -100.5316162, 309.5450439, -409.8053589, 407.8449707
3: -105.1281967, 399.3886414, -105.4614182, 402.5131226, -507.6412964, 504.8500366
4: -89.7864609, 363.2832031, -90.0528412, 365.8195190, -455.6059570, 453.3360596

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7985539, upper bound: 339.7982527
time: 1.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7957283, upper bound: 339.7970442
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -77.7812729, 263.4603271, -76.5251465, 258.0331726, -335.8143921, 339.9854431
1: -109.1529846, 261.0677795, -107.2658615, 255.8545990, -365.0075684, 368.3335876
2: -92.4850464, 287.5709229, -90.9418640, 281.8870544, -374.3721008, 378.5127869
3: -97.2033539, 374.3854675, -95.5396729, 366.7406006, -463.9439697, 469.9251404
4: -82.9996109, 340.1309814, -81.6038895, 333.3200378, -416.3196411, 421.7348633

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7879428, upper bound: 339.7676629
time: 1.10 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7854809, upper bound: 339.7680182
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -77.7812729, 263.4603271, -84.2214584, 281.0291443, -358.8103943, 347.6817627
1: -109.1529846, 261.0677795, -118.1812286, 278.9996948, -388.1526489, 379.2489624
2: -92.4850464, 287.5709229, -100.2603912, 307.3133545, -399.7983704, 387.8312988
3: -97.2033539, 374.3854675, -105.1281967, 399.3886414, -496.5919800, 479.5136719
4: -82.9996109, 340.1309814, -89.7864609, 363.2832031, -446.2828064, 429.9174500

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7879428, upper bound: 339.7953729
time: 1.20 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7854809, upper bound: 339.7957283
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -84.3763123, 283.3912354, -76.5251465, 258.0331726, -342.4094849, 359.9163818
1: -118.5572128, 281.0944824, -107.2658615, 255.8545990, -374.4118042, 388.3603210
2: -100.5316162, 309.5450439, -90.9418640, 281.8870544, -382.4186707, 400.4869080
3: -105.4614182, 402.5131226, -95.5396729, 366.7406006, -472.2020264, 498.0527344
4: -90.0528412, 365.8195190, -81.6038895, 333.3200378, -423.3728638, 447.4234009

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7965808, upper bound: 339.7701921
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7928889, upper bound: 339.7697890
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -84.3763123, 283.3912354, -84.2214584, 281.0291443, -365.4054565, 367.6127014
1: -118.5572128, 281.0944824, -118.1812286, 278.9996948, -397.5569153, 399.2756958
2: -100.5316162, 309.5450439, -100.2603912, 307.3133545, -407.8449707, 409.8053894
3: -105.4614182, 402.5131226, -105.1281967, 399.3886414, -504.8500366, 507.6412659
4: -90.0528412, 365.8195190, -89.7864609, 363.2832031, -453.3360596, 455.6059570

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7965808, upper bound: 339.7977921
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7928889, upper bound: 339.7954097
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -77.7812729, 263.4603271, -77.7812729, 263.4603271, -341.2415161, 341.2414551
1: -109.1529846, 261.0677795, -109.1529846, 261.0677795, -370.2207336, 370.2207336
2: -92.4850464, 287.5709229, -92.4850464, 287.5709229, -380.0559387, 380.0559387
3: -97.2033539, 374.3854675, -97.2033539, 374.3854675, -471.5888062, 471.5888062
4: -82.9996109, 340.1309814, -82.9996109, 340.1309814, -423.1305847, 423.1305847

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7886892, upper bound: 339.7858720
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7862273, upper bound: 339.7859224
time: 1.42 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -77.7812729, 263.4603271, -84.3763123, 283.3912354, -361.1725159, 347.8366394
1: -109.1529846, 261.0677795, -118.5572128, 281.0944824, -390.2474365, 379.6250000
2: -92.4850464, 287.5709229, -100.5316162, 309.5450439, -402.0300598, 388.1025391
3: -97.2033539, 374.3854675, -105.4614182, 402.5131226, -499.7164612, 479.8468933
4: -82.9996109, 340.1309814, -90.0528412, 365.8195190, -448.8191223, 430.1838379

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7886892, upper bound: 339.7932799
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7862273, upper bound: 339.7936353
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -84.3763123, 283.3912354, -77.7812729, 263.4603271, -347.8366089, 361.1725159
1: -118.5572128, 281.0944824, -109.1529846, 261.0677795, -379.6250000, 390.2474365
2: -100.5316162, 309.5450439, -92.4850464, 287.5709229, -388.1025391, 402.0300598
3: -105.4614182, 402.5131226, -97.2033539, 374.3854675, -479.8468933, 499.7164612
4: -90.0528412, 365.8195190, -82.9996109, 340.1309814, -430.1838379, 448.8191223

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7965808, upper bound: 339.7867655
time: 1.29 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7928889, upper bound: 339.7859926
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -84.3763123, 283.3912354, -84.3763123, 283.3912354, -367.7675476, 367.7675476
1: -118.5572128, 281.0944824, -118.5572128, 281.0944824, -399.6517029, 399.6517029
2: -100.5316162, 309.5450439, -100.5316162, 309.5450439, -410.0766602, 410.0766602
3: -105.4614182, 402.5131226, -105.4614182, 402.5131226, -507.9745178, 507.9745178
4: -90.0528412, 365.8195190, -90.0528412, 365.8195190, -455.8723755, 455.8723755

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7965808, upper bound: 339.7958177
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7928889, upper bound: 339.7954060
time: 1.01 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.72 seconds
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7631526, upper bound: 339.7530758
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7557520, upper bound: 339.7519158
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7978075, upper bound: 339.7726356
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7949819, upper bound: 339.7714272
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7978076, upper bound: 339.8003457
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7949819, upper bound: 339.7991372
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7579234, upper bound: 339.7746415
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7537583, upper bound: 339.7738271
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7579234, upper bound: 339.7840502
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7537583, upper bound: 339.7832358
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7985539, upper bound: 339.7908447
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7957283, upper bound: 339.7896363
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7985539, upper bound: 339.7982527
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7957283, upper bound: 339.7970442
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7879428, upper bound: 339.7676629
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7854809, upper bound: 339.7680182
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7879428, upper bound: 339.7953729
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7854809, upper bound: 339.7957283
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7965808, upper bound: 339.7701921
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7928889, upper bound: 339.7697890
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7965808, upper bound: 339.7977921
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7928889, upper bound: 339.7954097
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7886892, upper bound: 339.7858720
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7862273, upper bound: 339.7859224
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7886892, upper bound: 339.7932799
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7862273, upper bound: 339.7936353
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7965808, upper bound: 339.7867655
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7928889, upper bound: 339.7859926
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7965808, upper bound: 339.7958177
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -339.7928889, upper bound: 339.7954060

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -81.0358963, 269.8756104, -76.5251465, 258.0331726, -339.0690613, 346.4007568
1: -113.6971054, 267.9498291, -107.2658615, 255.8545990, -369.5516968, 375.2156677
2: -96.4887772, 295.1441040, -90.9418640, 281.8870544, -378.3758240, 386.0859680
3: -101.1333313, 383.3225708, -95.5396729, 366.7406006, -467.8739319, 478.8622131
4: -86.4139786, 348.7895813, -81.6038895, 333.3200378, -419.7339783, 430.3934631

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7961748, upper bound: 339.7644065
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7880319, upper bound: 339.7637527
time: 1.47 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7873232, upper bound: 339.7595876
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -88.0888672, 292.2174377, -76.5251465, 258.0331726, -346.1220093, 368.7425842
1: -123.6191406, 290.1853027, -107.2658615, 255.8545990, -379.4737549, 397.4511108
2: -104.8960876, 319.7673645, -90.9418640, 281.8870544, -386.7831421, 410.7092285
3: -109.8945084, 414.7697754, -95.5396729, 366.7406006, -476.6351013, 510.3094482
4: -93.7987595, 377.7793884, -81.6038895, 333.3200378, -427.1188049, 459.3832703

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7528858, upper bound: 339.7631526
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7519158, upper bound: 339.7557520
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -81.0358963, 269.8756104, -84.2214584, 281.0291443, -362.0650330, 354.0970764
1: -113.6971054, 267.9498291, -118.1812286, 278.9996948, -392.6967773, 386.1310425
2: -96.4887772, 295.1441040, -100.2603912, 307.3133545, -403.8021240, 395.4044495
3: -101.1333313, 383.3225708, -105.1281967, 399.3886414, -500.5219727, 488.4507446
4: -86.4139786, 348.7895813, -89.7864609, 363.2832031, -449.6971436, 438.5760193

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7985026, upper bound: 339.7991372
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7985026, upper bound: 339.7991372
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -88.0888672, 292.2174377, -84.2214584, 281.0291443, -369.1180115, 376.4389038
1: -123.6191406, 290.1853027, -118.1812286, 278.9996948, -402.6188354, 408.3664856
2: -104.8960876, 319.7673645, -100.2603912, 307.3133545, -412.2093811, 420.0277100
3: -109.8945084, 414.7697754, -105.1281967, 399.3886414, -509.2831421, 519.8978882
4: -93.7987595, 377.7793884, -89.7864609, 363.2832031, -457.0819702, 467.5658264

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7985026, upper bound: 339.7991372
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7985026, upper bound: 339.7991372
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -68.7997971, 232.7159271, -77.7812729, 263.4603271, -332.2601318, 310.4971924
1: -96.5648880, 230.8061523, -109.1529846, 261.0677795, -357.6326599, 339.9591370
2: -81.8181915, 254.3222504, -92.4850464, 287.5709229, -369.3890991, 346.8072815
3: -86.0115204, 330.8748779, -97.2033539, 374.3854675, -460.3969727, 428.0782471
4: -73.4907990, 300.7060547, -82.9996109, 340.1309814, -413.6217651, 383.7056580

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7570002, upper bound: 339.7745357
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7579103, upper bound: 339.7718195
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -72.0914078, 243.8674316, -77.7812729, 263.4603271, -335.5516663, 321.6487122
1: -100.8317337, 241.7197113, -109.1529846, 261.0677795, -361.8994751, 350.8726807
2: -85.4771194, 266.3782654, -92.4850464, 287.5709229, -373.0480042, 358.8632812
3: -89.8558121, 346.8189087, -97.2033539, 374.3854675, -464.2412720, 444.0222473
4: -76.8516617, 315.0495605, -82.9996109, 340.1309814, -416.9826355, 398.0491638

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7528351, upper bound: 339.7738271
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7537453, upper bound: 339.7711108
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -68.7997971, 232.7159271, -84.3763123, 283.3912354, -352.1910400, 317.0922241
1: -96.5648880, 230.8061523, -118.5572128, 281.0944824, -377.6593628, 349.3633728
2: -81.8181915, 254.3222504, -100.5316162, 309.5450439, -391.3632202, 354.8538818
3: -86.0115204, 330.8748779, -105.4614182, 402.5131226, -488.5246277, 436.3363037
4: -73.4907990, 300.7060547, -90.0528412, 365.8195190, -439.3103027, 390.7589111

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7597121, upper bound: 339.7839444
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7595137, upper bound: 339.7818934
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -72.0914078, 243.8674316, -84.3763123, 283.3912354, -355.4826355, 328.2437439
1: -100.8317337, 241.7197113, -118.5572128, 281.0944824, -381.9261780, 360.2769165
2: -85.4771194, 266.3782654, -100.5316162, 309.5450439, -395.0220947, 366.9098816
3: -89.8558121, 346.8189087, -105.4614182, 402.5131226, -492.3689270, 452.2803040
4: -76.8516617, 315.0495605, -90.0528412, 365.8195190, -442.6711426, 405.1024170

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7555470, upper bound: 339.7832357
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7553487, upper bound: 339.7811847
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -81.0358963, 269.8756104, -77.7812729, 263.4603271, -344.4962158, 347.6568298
1: -113.6971054, 267.9498291, -109.1529846, 261.0677795, -374.7648926, 377.1028137
2: -96.4887772, 295.1441040, -92.4850464, 287.5709229, -384.0596924, 387.6291199
3: -101.1333313, 383.3225708, -97.2033539, 374.3854675, -475.5187988, 480.5258789
4: -86.4139786, 348.7895813, -82.9996109, 340.1309814, -426.5449219, 431.7891846

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7953729, upper bound: 339.7896363
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7953729, upper bound: 339.7896363
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -88.0888672, 292.2174377, -77.7812729, 263.4603271, -351.5491333, 369.9986572
1: -123.6191406, 290.1853027, -109.1529846, 261.0677795, -384.6869202, 399.3382568
2: -104.8960876, 319.7673645, -92.4850464, 287.5709229, -392.4669495, 412.2523804
3: -109.8945084, 414.7697754, -97.2033539, 374.3854675, -484.2799683, 511.9731445
4: -93.7987595, 377.7793884, -82.9996109, 340.1309814, -433.9297485, 460.7789917

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7953729, upper bound: 339.7896363
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7953729, upper bound: 339.7896363
time: 1.45 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -81.0358963, 269.8756104, -84.3763123, 283.3912354, -364.4271240, 354.2519226
1: -113.6971054, 267.9498291, -118.5572128, 281.0944824, -394.7915955, 386.5070496
2: -96.4887772, 295.1441040, -100.5316162, 309.5450439, -406.0338135, 395.6757202
3: -101.1333313, 383.3225708, -105.4614182, 402.5131226, -503.6464539, 488.7839355
4: -86.4139786, 348.7895813, -90.0528412, 365.8195190, -452.2334290, 438.8424072

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7974990, upper bound: 339.7970442
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7974990, upper bound: 339.7970442
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -88.0888672, 292.2174377, -84.3763123, 283.3912354, -371.4801025, 376.5937500
1: -123.6191406, 290.1853027, -118.5572128, 281.0944824, -404.7136230, 408.7425232
2: -104.8960876, 319.7673645, -100.5316162, 309.5450439, -414.4410706, 420.2989807
3: -109.8945084, 414.7697754, -105.4614182, 402.5131226, -512.4076538, 520.2311401
4: -93.7987595, 377.7793884, -90.0528412, 365.8195190, -459.6182556, 467.8322144

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7974990, upper bound: 339.7970442
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7974990, upper bound: 339.7970442
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -74.5017242, 252.0583954, -76.5251465, 258.0331726, -332.5349121, 328.5835571
1: -104.5777283, 249.7615204, -107.2658615, 255.8545990, -360.4323120, 357.0273438
2: -88.6153183, 275.1112366, -90.9418640, 281.8870544, -370.5023804, 366.0531006
3: -93.1252823, 358.0035706, -95.5396729, 366.7406006, -459.8658752, 453.5432434
4: -79.5323639, 325.3593140, -81.6038895, 333.3200378, -412.8524170, 406.9631958

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7861174, upper bound: 339.7590265
time: 1.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7745357, upper bound: 339.7570002
time: 1.28 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7738271, upper bound: 339.7528351
time: 1.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -82.1193466, 276.7013550, -76.5251465, 258.0331726, -340.1524658, 353.2265015
1: -115.4125595, 274.2441406, -107.2658615, 255.8545990, -371.2671509, 381.5099792
2: -97.7877274, 302.1891785, -90.9418640, 281.8870544, -379.6747742, 393.1310425
3: -102.6778336, 392.8051453, -95.5396729, 366.7406006, -469.4184265, 488.3447876
4: -87.5935593, 357.3958130, -81.6038895, 333.3200378, -420.9136047, 438.9996948

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7594418, upper bound: 339.7637479
time: 1.21 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7584719, upper bound: 339.7563473
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -74.5017242, 252.0583954, -84.2214584, 281.0291443, -355.5308838, 336.2798462
1: -104.5777283, 249.7615204, -118.1812286, 278.9996948, -383.5773621, 367.9426880
2: -88.6153183, 275.1112366, -100.2603912, 307.3133545, -395.9286499, 375.3715820
3: -93.1252823, 358.0035706, -105.1281967, 399.3886414, -492.5139160, 463.1317749
4: -79.5323639, 325.3593140, -89.7864609, 363.2832031, -442.8155518, 415.1457520

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7896363, upper bound: 339.7953729
time: 1.25 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7896363, upper bound: 339.7953729
time: 1.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -82.1193466, 276.7013550, -84.2214584, 281.0291443, -363.1484680, 360.9228210
1: -115.4125595, 274.2441406, -118.1812286, 278.9996948, -394.4122009, 392.4253540
2: -97.7877274, 302.1891785, -100.2603912, 307.3133545, -405.1010437, 402.4494629
3: -102.6778336, 392.8051453, -105.1281967, 399.3886414, -502.0664673, 497.9333191
4: -87.5935593, 357.3958130, -89.7864609, 363.2832031, -450.8767700, 447.1822510

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7896363, upper bound: 339.7957283
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7896363, upper bound: 339.7957283
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -81.1669083, 272.2276917, -76.5251465, 258.0331726, -339.1999512, 348.7528381
1: -114.0635605, 270.0139465, -107.2658615, 255.8545990, -369.9181519, 377.2798157
2: -96.7370834, 297.3360596, -90.9418640, 281.8870544, -378.6241455, 388.2779236
3: -101.4570694, 386.4553833, -95.5396729, 366.7406006, -468.1976624, 481.9950562
4: -86.6637650, 351.3312073, -81.6038895, 333.3200378, -419.9837952, 432.9350891

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7952290, upper bound: 339.7622699
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7839444, upper bound: 339.7597121
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7832357, upper bound: 339.7555470
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -88.2225113, 294.4195557, -76.5251465, 258.0331726, -346.2556458, 370.9447021
1: -123.9681091, 292.1536255, -107.2658615, 255.8545990, -379.8226929, 399.4194336
2: -105.1256409, 321.8575439, -90.9418640, 281.8870544, -387.0126953, 412.7994080
3: -110.1987991, 417.7736511, -95.5396729, 366.7406006, -476.9393921, 513.3132935
4: -94.0284424, 380.2252808, -81.6038895, 333.3200378, -427.3484802, 461.8291626

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7818933, upper bound: 339.7595137
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7811847, upper bound: 339.7553487
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -81.1669083, 272.2276917, -84.2214584, 281.0291443, -362.1959534, 356.4491577
1: -114.0635605, 270.0139465, -118.1812286, 278.9996948, -393.0632019, 388.1951599
2: -96.7370834, 297.3360596, -100.2603912, 307.3133545, -404.0503845, 397.5964355
3: -101.4570694, 386.4553833, -105.1281967, 399.3886414, -500.8457031, 491.5835876
4: -86.6637650, 351.3312073, -89.7864609, 363.2832031, -449.9469604, 441.1176453

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7954087, upper bound: 339.7954097
time: 1.10 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7954087, upper bound: 339.7954097
time: 1.36 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -88.2225113, 294.4195557, -84.2214584, 281.0291443, -369.2516479, 378.6410217
1: -123.9681091, 292.1536255, -118.1812286, 278.9996948, -402.9677429, 410.3348083
2: -105.1256409, 321.8575439, -100.2603912, 307.3133545, -412.4389954, 422.1178589
3: -110.1987991, 417.7736511, -105.1281967, 399.3886414, -509.5874329, 522.9017334
4: -94.0284424, 380.2252808, -89.7864609, 363.2832031, -457.3116455, 470.0117188

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7954087, upper bound: 339.7954097
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7954087, upper bound: 339.7954097
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -74.5017242, 252.0583954, -77.7812729, 263.4603271, -337.9620361, 329.8396606
1: -104.5777283, 249.7615204, -109.1529846, 261.0677795, -365.6454773, 358.9144592
2: -88.6153183, 275.1112366, -92.4850464, 287.5709229, -376.1861877, 367.5962524
3: -93.1252823, 358.0035706, -97.2033539, 374.3854675, -467.5107422, 455.2069092
4: -79.5323639, 325.3593140, -82.9996109, 340.1309814, -419.6633301, 408.3589172

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7858688, upper bound: 339.7858720
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7858688, upper bound: 339.7858720
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -82.1193466, 276.7013550, -77.7812729, 263.4603271, -345.5795593, 354.4826050
1: -115.4125595, 274.2441406, -109.1529846, 261.0677795, -376.4803162, 383.3971252
2: -97.7877274, 302.1891785, -92.4850464, 287.5709229, -385.3586121, 394.6741638
3: -102.6778336, 392.8051453, -97.2033539, 374.3854675, -477.0632935, 490.0084839
4: -87.5935593, 357.3958130, -82.9996109, 340.1309814, -427.7245483, 440.3954163

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7858688, upper bound: 339.7859224
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7858688, upper bound: 339.7859224
time: 1.25 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -74.5017242, 252.0583954, -84.3763123, 283.3912354, -357.8929443, 336.4346924
1: -104.5777283, 249.7615204, -118.5572128, 281.0944824, -385.6722107, 368.3187256
2: -88.6153183, 275.1112366, -100.5316162, 309.5450439, -398.1603394, 375.6428528
3: -93.1252823, 358.0035706, -105.4614182, 402.5131226, -495.6383972, 463.4649963
4: -79.5323639, 325.3593140, -90.0528412, 365.8195190, -445.3518677, 415.4121704

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7879981, upper bound: 339.7932799
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7879981, upper bound: 339.7932799
time: 1.21 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -82.1193466, 276.7013550, -84.3763123, 283.3912354, -365.5105896, 361.0776672
1: -115.4125595, 274.2441406, -118.5572128, 281.0944824, -396.5070496, 392.8013611
2: -97.7877274, 302.1891785, -100.5316162, 309.5450439, -407.3327332, 402.7207947
3: -102.6778336, 392.8051453, -105.4614182, 402.5131226, -505.1909485, 498.2665710
4: -87.5935593, 357.3958130, -90.0528412, 365.8195190, -453.4130554, 447.4486694

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7879981, upper bound: 339.7936353
time: 1.22 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7879981, upper bound: 339.7936353
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -81.1669083, 272.2276917, -77.7812729, 263.4603271, -344.6271057, 350.0089111
1: -114.0635605, 270.0139465, -109.1529846, 261.0677795, -375.1313171, 379.1669312
2: -96.7370834, 297.3360596, -92.4850464, 287.5709229, -384.3079529, 389.8211060
3: -101.4570694, 386.4553833, -97.2033539, 374.3854675, -475.8425293, 483.6587524
4: -86.6637650, 351.3312073, -82.9996109, 340.1309814, -426.7947388, 434.3308105

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7867957, upper bound: 339.7859926
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7867957, upper bound: 339.7859926
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -88.2225113, 294.4195557, -77.7812729, 263.4603271, -351.6827393, 372.2007751
1: -123.9681091, 292.1536255, -109.1529846, 261.0677795, -385.0358276, 401.3065796
2: -105.1256409, 321.8575439, -92.4850464, 287.5709229, -392.6965637, 414.3425598
3: -110.1987991, 417.7736511, -97.2033539, 374.3854675, -484.5842590, 514.9769897
4: -94.0284424, 380.2252808, -82.9996109, 340.1309814, -434.1594238, 463.2248840

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7867957, upper bound: 339.7859926
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7867957, upper bound: 339.7859926
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -81.1669083, 272.2276917, -84.3763123, 283.3912354, -364.5580750, 356.6040039
1: -114.0635605, 270.0139465, -118.5572128, 281.0944824, -395.1580505, 388.5711670
2: -96.7370834, 297.3360596, -100.5316162, 309.5450439, -406.2820740, 397.8676758
3: -101.4570694, 386.4553833, -105.4614182, 402.5131226, -503.9701843, 491.9168091
4: -86.6637650, 351.3312073, -90.0528412, 365.8195190, -452.4832764, 441.3840332

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7954060, upper bound: 339.7954060
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7954060, upper bound: 339.7954060
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -88.2225113, 294.4195557, -84.3763123, 283.3912354, -371.6137390, 378.7958679
1: -123.9681091, 292.1536255, -118.5572128, 281.0944824, -405.0625305, 410.7108459
2: -105.1256409, 321.8575439, -100.5316162, 309.5450439, -414.6706848, 422.3891602
3: -110.1987991, 417.7736511, -105.4614182, 402.5131226, -512.7119141, 523.2349243
4: -94.0284424, 380.2252808, -90.0528412, 365.8195190, -459.8479614, 470.2781372

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7954060, upper bound: 339.7954060
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7954060, upper bound: 339.7954060
time: 1.21 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.82 seconds
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7880319, upper bound: 339.7637527
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7873232, upper bound: 339.7595876
IS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7528858, upper bound: 339.7631526
IS_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7519158, upper bound: 339.7557520
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7985026, upper bound: 339.7991372
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7985026, upper bound: 339.7991372
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7985026, upper bound: 339.7991372
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7985026, upper bound: 339.7991372
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7570002, upper bound: 339.7745357
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7579103, upper bound: 339.7718195
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7528351, upper bound: 339.7738271
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7537453, upper bound: 339.7711108
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7597121, upper bound: 339.7839444
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7595137, upper bound: 339.7818934
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7555470, upper bound: 339.7832357
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7553487, upper bound: 339.7811847
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7953729, upper bound: 339.7896363
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7953729, upper bound: 339.7896363
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7953729, upper bound: 339.7896363
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7953729, upper bound: 339.7896363
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7974990, upper bound: 339.7970442
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7974990, upper bound: 339.7970442
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7974990, upper bound: 339.7970442
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7974990, upper bound: 339.7970442
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7745357, upper bound: 339.7570002
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7738271, upper bound: 339.7528351
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7594418, upper bound: 339.7637479
IS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7584719, upper bound: 339.7563473
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7896363, upper bound: 339.7953729
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7896363, upper bound: 339.7953729
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7896363, upper bound: 339.7957283
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7896363, upper bound: 339.7957283
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7839444, upper bound: 339.7597121
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7832357, upper bound: 339.7555470
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7818933, upper bound: 339.7595137
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7811847, upper bound: 339.7553487
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7954087, upper bound: 339.7954097
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7954087, upper bound: 339.7954097
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7954087, upper bound: 339.7954097
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7954087, upper bound: 339.7954097
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7858688, upper bound: 339.7858720
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7858688, upper bound: 339.7858720
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7858688, upper bound: 339.7859224
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7858688, upper bound: 339.7859224
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7879981, upper bound: 339.7932799
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7879981, upper bound: 339.7932799
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7879981, upper bound: 339.7936353
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7879981, upper bound: 339.7936353
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7867957, upper bound: 339.7859926
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7867957, upper bound: 339.7859926
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7867957, upper bound: 339.7859926
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7867957, upper bound: 339.7859926
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7954060, upper bound: 339.7954060
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7954060, upper bound: 339.7954060
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7954060, upper bound: 339.7954060
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.82
Output dim: 0, lower bound: -339.7954060, upper bound: 339.7954060

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -81.0358963, 269.8756104, -68.7997971, 232.7159271, -313.7518311, 338.6754150
1: -113.6971054, 267.9498291, -96.5648880, 230.8061523, -344.5032654, 364.5147095
2: -96.4887772, 295.1441040, -81.8181915, 254.3222504, -350.8110352, 376.9622803
3: -101.1333313, 383.3225708, -86.0115204, 330.8748779, -432.0082092, 469.3340759
4: -86.4139786, 348.7895813, -73.4907990, 300.7060547, -387.1199646, 422.2803955

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7873232, upper bound: 339.7595876
time: 1.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7873232, upper bound: 339.7595876
time: 1.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -81.0358963, 269.8756104, -72.0914078, 243.8674316, -324.9033203, 341.9669800
1: -113.6971054, 267.9498291, -100.8317337, 241.7197113, -355.4168091, 368.7815552
2: -96.4887772, 295.1441040, -85.4771194, 266.3782654, -362.8670349, 380.6211548
3: -101.1333313, 383.3225708, -89.8558121, 346.8189087, -447.9522400, 473.1783752
4: -86.4139786, 348.7895813, -76.8516617, 315.0495605, -401.4635315, 425.6412048

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7873232, upper bound: 339.7595876
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7873232, upper bound: 339.7595876
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -81.0358963, 269.8756104, -81.0358963, 269.8756104, -350.9114990, 350.9114990
1: -113.6971054, 267.9498291, -113.6971054, 267.9498291, -381.6469421, 381.6469421
2: -96.4887772, 295.1441040, -96.4887772, 295.1441040, -391.6328735, 391.6328735
3: -101.1333313, 383.3225708, -101.1333313, 383.3225708, -484.4559021, 484.4559021
4: -86.4139786, 348.7895813, -86.4139786, 348.7895813, -435.2034912, 435.2034912

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8006166, upper bound: 339.7921652
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7924428, upper bound: 339.7914198
time: 1.32 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -81.0358963, 269.8756104, -88.0888672, 292.2174377, -373.2533264, 357.9644775
1: -113.6971054, 267.9498291, -123.6191406, 290.1853027, -403.8823853, 391.5689697
2: -96.4887772, 295.1441040, -104.8960876, 319.7673645, -416.2561340, 400.0401306
3: -101.1333313, 383.3225708, -109.8945084, 414.7697754, -515.9030151, 493.2170715
4: -86.4139786, 348.7895813, -93.7987595, 377.7793884, -464.1932983, 442.5883484

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8006166, upper bound: 339.7921652
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7924428, upper bound: 339.7914197
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -88.0888672, 292.2174377, -81.0358963, 269.8756104, -357.9644775, 373.2533264
1: -123.6191406, 290.1853027, -113.6971054, 267.9498291, -391.5689697, 403.8823853
2: -104.8960876, 319.7673645, -96.4887772, 295.1441040, -400.0401306, 416.2561340
3: -109.8945084, 414.7697754, -101.1333313, 383.3225708, -493.2170715, 515.9030762
4: -93.7987595, 377.7793884, -86.4139786, 348.7895813, -442.5883179, 464.1932983

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7962362, upper bound: 339.7909158
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7903603, upper bound: 339.7908134
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -88.0888672, 292.2174377, -88.0888672, 292.2174377, -380.3063049, 380.3063049
1: -123.6191406, 290.1853027, -123.6191406, 290.1853027, -413.8044434, 413.8044434
2: -104.8960876, 319.7673645, -104.8960876, 319.7673645, -424.6633911, 424.6633911
3: -109.8945084, 414.7697754, -109.8945084, 414.7697754, -524.6643066, 524.6643066
4: -93.7987595, 377.7793884, -93.7987595, 377.7793884, -471.5781250, 471.5781555

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7962362, upper bound: 339.7909158
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7903603, upper bound: 339.7908134
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -68.7997971, 232.7159271, -74.5017242, 252.0583954, -320.8581848, 307.2176514
1: -96.5648880, 230.8061523, -104.5777283, 249.7615204, -346.3263855, 335.3838806
2: -81.8181915, 254.3222504, -88.6153183, 275.1112366, -356.9294434, 342.9375610
3: -86.0115204, 330.8748779, -93.1252823, 358.0035706, -444.0150757, 424.0001526
4: -73.4907990, 300.7060547, -79.5323639, 325.3593140, -398.8500977, 380.2384033

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7390893, upper bound: 339.7707068
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7570002, upper bound: 339.7745357
time: 1.50 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -68.7997971, 232.7159271, -82.1193466, 276.7013550, -345.5011597, 314.8352661
1: -96.5648880, 230.8061523, -115.4125595, 274.2441406, -370.8090210, 346.2187195
2: -81.8181915, 254.3222504, -97.7877274, 302.1891785, -384.0073242, 352.1099548
3: -86.0115204, 330.8748779, -102.6778336, 392.8051453, -478.8166504, 433.5527039
4: -73.4907990, 300.7060547, -87.5935593, 357.3958130, -430.8865967, 388.2995911

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7546485, upper bound: 339.7479521
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7396345, upper bound: 339.7461218
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -72.0914078, 243.8674316, -74.5017242, 252.0583954, -324.1498108, 318.3691406
1: -100.8317337, 241.7197113, -104.5777283, 249.7615204, -350.5932007, 346.2974243
2: -85.4771194, 266.3782654, -88.6153183, 275.1112366, -360.5883179, 354.9935608
3: -89.8558121, 346.8189087, -93.1252823, 358.0035706, -447.8593750, 439.9441833
4: -76.8516617, 315.0495605, -79.5323639, 325.3593140, -402.2109375, 394.5819092

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7075076, upper bound: 339.7687795
time: 1.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7528351, upper bound: 339.7738271
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -72.0914078, 243.8674316, -82.1193466, 276.7013550, -348.7927246, 325.9867859
1: -100.8317337, 241.7197113, -115.4125595, 274.2441406, -375.0758667, 357.1322632
2: -85.4771194, 266.3782654, -97.7877274, 302.1891785, -387.6661987, 364.1659546
3: -89.8558121, 346.8189087, -102.6778336, 392.8051453, -482.6609497, 449.4967346
4: -76.8516617, 315.0495605, -87.5935593, 357.3958130, -434.2474365, 402.6431274

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7520238, upper bound: 339.7472435
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7456043, upper bound: 339.7466458
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -68.7997971, 232.7159271, -81.1669083, 272.2276917, -341.0274963, 313.8827820
1: -96.5648880, 230.8061523, -114.0635605, 270.0139465, -366.5788269, 344.8697205
2: -81.8181915, 254.3222504, -96.7370834, 297.3360596, -379.1542358, 351.0592957
3: -86.0115204, 330.8748779, -101.4570694, 386.4553833, -472.4669189, 432.3319397
4: -73.4907990, 300.7060547, -86.6637650, 351.3312073, -424.8220215, 387.3698120

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7408813, upper bound: 339.7733183
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7587922, upper bound: 339.7771473
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -68.7997971, 232.7159271, -88.2225113, 294.4195557, -363.2193604, 320.9384460
1: -96.5648880, 230.8061523, -123.9681091, 292.1536255, -388.7185059, 354.7742310
2: -81.8181915, 254.3222504, -105.1256409, 321.8575439, -403.6757202, 359.4478760
3: -86.0115204, 330.8748779, -110.1987991, 417.7736511, -503.7851562, 441.0736694
4: -73.4907990, 300.7060547, -94.0284424, 380.2252808, -453.7160645, 394.7344971

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7411601, upper bound: 339.7728384
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7590711, upper bound: 339.7766673
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -72.0914078, 243.8674316, -81.1669083, 272.2276917, -344.3190613, 325.0343323
1: -100.8317337, 241.7197113, -114.0635605, 270.0139465, -370.8456726, 355.7832642
2: -85.4771194, 266.3782654, -96.7370834, 297.3360596, -382.8131409, 363.1152954
3: -89.8558121, 346.8189087, -101.4570694, 386.4553833, -476.3111877, 448.2759705
4: -76.8516617, 315.0495605, -86.6637650, 351.3312073, -428.1828308, 401.7133179

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7102194, upper bound: 339.7782266
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7555470, upper bound: 339.7832357
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -72.0914078, 243.8674316, -88.2225113, 294.4195557, -366.5109253, 332.0899353
1: -100.8317337, 241.7197113, -123.9681091, 292.1536255, -392.9853210, 365.6878052
2: -85.4771194, 266.3782654, -105.1256409, 321.8575439, -407.3345947, 371.5039062
3: -89.8558121, 346.8189087, -110.1987991, 417.7736511, -507.6294556, 457.0177002
4: -76.8516617, 315.0495605, -94.0284424, 380.2252808, -457.0769043, 409.0780029

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7100211, upper bound: 339.7761756
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7553487, upper bound: 339.7811847
time: 1.17 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -81.0358963, 269.8756104, -74.5017242, 252.0583954, -333.0942993, 344.3773193
1: -113.6971054, 267.9498291, -104.5777283, 249.7615204, -363.4586182, 372.5275269
2: -96.4887772, 295.1441040, -88.6153183, 275.1112366, -371.6000061, 383.7593994
3: -101.1333313, 383.3225708, -93.1252823, 358.0035706, -459.1369019, 476.4478455
4: -86.4139786, 348.7895813, -79.5323639, 325.3593140, -411.7732239, 428.3219604

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7956644, upper bound: 339.7776906
time: 1.28 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7864051, upper bound: 339.7769451
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -81.0358963, 269.8756104, -82.1193466, 276.7013550, -357.7372437, 351.9949341
1: -113.6971054, 267.9498291, -115.4125595, 274.2441406, -387.9412537, 383.3623962
2: -96.4887772, 295.1441040, -97.7877274, 302.1891785, -398.6779175, 392.9317932
3: -101.1333313, 383.3225708, -102.6778336, 392.8051453, -493.9384766, 486.0003967
4: -86.4139786, 348.7895813, -87.5935593, 357.3958130, -443.8097229, 436.3831177

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7956644, upper bound: 339.7776906
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7864051, upper bound: 339.7769451
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -88.0888672, 292.2174377, -74.5017242, 252.0583954, -340.1472778, 366.7191772
1: -123.6191406, 290.1853027, -104.5777283, 249.7615204, -373.3806763, 394.7629700
2: -104.8960876, 319.7673645, -88.6153183, 275.1112366, -380.0072632, 408.3826599
3: -109.8945084, 414.7697754, -93.1252823, 358.0035706, -467.8980713, 507.8950500
4: -93.7987595, 377.7793884, -79.5323639, 325.3593140, -419.1580811, 457.3117676

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7764412
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7846673, upper bound: 339.7763388
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -88.0888672, 292.2174377, -82.1193466, 276.7013550, -364.7902222, 374.3367615
1: -123.6191406, 290.1853027, -115.4125595, 274.2441406, -397.8632812, 405.5978394
2: -104.8960876, 319.7673645, -97.7877274, 302.1891785, -407.0851746, 417.5550537
3: -109.8945084, 414.7697754, -102.6778336, 392.8051453, -502.6996460, 517.4475708
4: -93.7987595, 377.7793884, -87.5935593, 357.3958130, -451.1945801, 465.3729248

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7922315, upper bound: 339.7764412
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7846673, upper bound: 339.7763388
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -81.0358963, 269.8756104, -81.1669083, 272.2276917, -353.2635803, 351.0424500
1: -113.6971054, 267.9498291, -114.0635605, 270.0139465, -383.7110596, 382.0133667
2: -96.4887772, 295.1441040, -96.7370834, 297.3360596, -393.8248291, 391.8811340
3: -101.1333313, 383.3225708, -101.4570694, 386.4553833, -487.5887146, 484.7796326
4: -86.4139786, 348.7895813, -86.6637650, 351.3312073, -437.7451477, 435.4533386

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7981780, upper bound: 339.7877645
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7889186, upper bound: 339.7870190
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -81.0358963, 269.8756104, -88.2225113, 294.4195557, -375.4554443, 358.0981140
1: -113.6971054, 267.9498291, -123.9681091, 292.1536255, -405.8507385, 391.9178772
2: -96.4887772, 295.1441040, -105.1256409, 321.8575439, -418.3463135, 400.2697449
3: -101.1333313, 383.3225708, -110.1987991, 417.7736511, -518.9068604, 493.5213623
4: -86.4139786, 348.7895813, -94.0284424, 380.2252808, -466.6391907, 442.8180237

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7981780, upper bound: 339.7877645
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7889186, upper bound: 339.7870190
time: 1.27 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -88.0888672, 292.2174377, -81.1669083, 272.2276917, -360.3165283, 373.3842773
1: -123.6191406, 290.1853027, -114.0635605, 270.0139465, -393.6330872, 404.2488098
2: -104.8960876, 319.7673645, -96.7370834, 297.3360596, -402.2321472, 416.5043945
3: -109.8945084, 414.7697754, -101.4570694, 386.4553833, -496.3498840, 516.2268677
4: -93.7987595, 377.7793884, -86.6637650, 351.3312073, -445.1299744, 464.4431458

Time for backsubstitution: 2.47 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=385.80084228515625
rel_dist={0: [-339.80568762378186, 339.805687623782]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042640, upper bound: 339.8053471
time: 1.17 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042151, upper bound: 339.8042151
time: 0.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.20 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.20
Output dim: 0, lower bound: -339.8042640, upper bound: 339.8053471
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.20
Output dim: 0, lower bound: -339.8042151, upper bound: 339.8042151

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -88.4568100, 296.2978516, -381.3928223, 372.6272888
1: -119.3920059, 282.0884705, -124.1144333, 294.0228577, -413.4148254, 406.2029114
2: -101.2751236, 310.7092590, -105.2662048, 323.7992249, -425.0743408, 415.9754028
3: -106.2089005, 403.8296509, -110.4195175, 420.8130493, -527.0219727, 514.2490845
4: -90.6926956, 367.2429504, -94.2557297, 382.5349121, -473.2276001, 461.4986877

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042151, upper bound: 339.8042151
time: 1.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042151, upper bound: 339.8042151
time: 0.83 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -85.5067825, 287.2420959, -88.6961975, 297.1046753, -382.6114502, 375.9382324
1: -120.1114273, 284.9082947, -124.4471970, 294.8176575, -414.9290466, 409.3554993
2: -101.8361511, 313.7402954, -105.5478058, 324.6724243, -426.5085449, 419.2880859
3: -106.8467102, 407.9730225, -110.7164154, 421.9519958, -528.7987061, 518.6894531
4: -91.2148666, 370.7578735, -94.5076294, 383.5692749, -474.7841492, 465.2655029

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042151, upper bound: 339.8042151
time: 0.98 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042151, upper bound: 339.8042151
time: 1.25 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.76 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.76
Output dim: 0, lower bound: -339.8042151, upper bound: 339.8042151
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.76
Output dim: 0, lower bound: -339.8042151, upper bound: 339.8042151
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.76
Output dim: 0, lower bound: -339.8042151, upper bound: 339.8042151
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.76
Output dim: 0, lower bound: -339.8042151, upper bound: 339.8042151

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -85.0949707, 284.1704712, -369.2654419, 369.2654114
1: -119.3920059, 282.0884705, -119.3920059, 282.0884705, -401.4804077, 401.4804077
2: -101.2751236, 310.7092590, -101.2751236, 310.7092590, -411.9843445, 411.9843445
3: -106.2089005, 403.8296509, -106.2089005, 403.8296509, -510.0385437, 510.0385437
4: -90.6926956, 367.2429504, -90.6926956, 367.2429504, -457.9356384, 457.9356384

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7699983, upper bound: 339.7923764
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8024352, upper bound: 339.8027842
time: 1.25 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -85.5067825, 287.2420959, -372.3370361, 369.6772461
1: -119.3920059, 282.0884705, -120.1114273, 284.9082947, -404.3002319, 402.1998596
2: -101.2751236, 310.7092590, -101.8361511, 313.7402954, -415.0153809, 412.5454102
3: -106.2089005, 403.8296509, -106.8467102, 407.9730225, -514.1819458, 510.6763611
4: -90.6926956, 367.2429504, -91.2148666, 370.7578735, -461.4505615, 458.4578247

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7699983, upper bound: 339.7923764
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8024352, upper bound: 339.8027842
time: 0.95 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -85.5067825, 287.2420959, -85.0949707, 284.1704712, -369.6772461, 372.3370361
1: -120.1114273, 284.9082947, -119.3920059, 282.0884705, -402.1998596, 404.3002319
2: -101.8361511, 313.7402954, -101.2751236, 310.7092590, -412.5454102, 415.0153809
3: -106.8467102, 407.9730225, -106.2089005, 403.8296509, -510.6763611, 514.1819458
4: -91.2148666, 370.7578735, -90.6926956, 367.2429504, -458.4578247, 461.4505310

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7915392, upper bound: 339.7986506
time: 1.49 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8011052, upper bound: 339.8011052
time: 1.02 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -85.5067825, 287.2420959, -85.5067825, 287.2420959, -372.7488708, 372.7488708
1: -120.1114273, 284.9082947, -120.1114273, 284.9082947, -405.0196838, 405.0196838
2: -101.8361511, 313.7402954, -101.8361511, 313.7402954, -415.5764465, 415.5764465
3: -106.8467102, 407.9730225, -106.8467102, 407.9730225, -514.8197021, 514.8197021
4: -91.2148666, 370.7578735, -91.2148666, 370.7578735, -461.9727173, 461.9727173

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7915392, upper bound: 339.7986506
time: 1.43 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8011052, upper bound: 339.8011052
time: 0.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.95 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.95
Output dim: 0, lower bound: -339.7699983, upper bound: 339.7923764
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.95
Output dim: 0, lower bound: -339.8024352, upper bound: 339.8027842
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.95
Output dim: 0, lower bound: -339.7699983, upper bound: 339.7923764
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.95
Output dim: 0, lower bound: -339.8024352, upper bound: 339.8027842
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.95
Output dim: 0, lower bound: -339.7915392, upper bound: 339.7986506
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.95
Output dim: 0, lower bound: -339.8011052, upper bound: 339.8011052
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.95
Output dim: 0, lower bound: -339.7915392, upper bound: 339.7986506
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.95
Output dim: 0, lower bound: -339.8011052, upper bound: 339.8011052

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -76.5251465, 258.0331726, -84.2484131, 281.5357971, -358.0609436, 342.2815552
1: -107.2658615, 255.8545990, -118.1968994, 279.4552002, -386.7210083, 374.0515137
2: -90.9418640, 281.8870544, -100.2553787, 307.8198547, -398.7617188, 382.1424255
3: -95.5396729, 366.7406006, -105.1548920, 400.1231079, -495.6627808, 471.8954468
4: -81.6038895, 333.3200378, -89.7930832, 363.8727722, -445.4766541, 423.1131287

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7674618, upper bound: 339.7674618
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7674618, upper bound: 339.7965074
time: 1.65 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -84.2214584, 281.0291443, -85.0485764, 284.0029297, -368.2243958, 366.0776672
1: -118.1812286, 278.9996948, -119.3276138, 281.9238892, -400.1051025, 398.3272705
2: -100.2603912, 307.3133545, -101.2211838, 310.5284119, -410.7887573, 408.5345154
3: -105.1281967, 399.3886414, -106.1514359, 403.5927734, -508.7209778, 505.5400696
4: -89.7864609, 363.2832031, -90.6445084, 367.0319824, -456.8184204, 453.9277039

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7965074, upper bound: 339.7722590
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7965074, upper bound: 339.8045731
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -76.5251465, 258.0331726, -84.6927872, 284.6872253, -361.2123718, 342.7259216
1: -107.2658615, 255.8545990, -118.9581451, 282.3569946, -389.6228638, 374.8127441
2: -90.9418640, 281.8870544, -100.8533630, 310.9400940, -401.8819580, 382.7404175
3: -95.5396729, 366.7406006, -105.8310165, 404.3847656, -499.9244080, 472.5715637
4: -81.6038895, 333.3200378, -90.3477631, 367.4818726, -449.0857544, 423.6677551

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7683776, upper bound: 339.7850231
time: 1.39 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7683776, upper bound: 339.7923764
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -84.2214584, 281.0291443, -85.4343643, 286.9955750, -371.2170410, 366.4635010
1: -118.1812286, 278.9996948, -120.0115356, 284.6641541, -402.8453674, 399.0111694
2: -100.2603912, 307.3133545, -101.7520218, 313.4716492, -413.7319946, 409.0653687
3: -105.1281967, 399.3886414, -106.7577209, 407.6237488, -512.7519531, 506.1463318
4: -89.7864609, 363.2832031, -91.1399841, 370.4416504, -460.2280884, 454.4231873

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7999904, upper bound: 339.7931278
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7999904, upper bound: 339.8027842
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -77.7812729, 263.4603271, -84.2484131, 281.5357971, -359.3170776, 347.7086487
1: -109.1529846, 261.0677795, -118.1968994, 279.4552002, -388.6081543, 379.2646484
2: -92.4850464, 287.5709229, -100.2553787, 307.8198547, -400.3048706, 387.8262939
3: -97.2033539, 374.3854675, -105.1548920, 400.1231079, -497.3264465, 479.5403137
4: -82.9996109, 340.1309814, -89.7930832, 363.8727722, -446.8723450, 429.9240723

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7850230, upper bound: 339.7683776
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7850230, upper bound: 339.7999904
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -84.3763123, 283.3912354, -85.0485764, 284.0029297, -368.3792419, 368.4397888
1: -118.5572128, 281.0944824, -119.3276138, 281.9238892, -400.4811096, 400.4220886
2: -100.5316162, 309.5450439, -101.2211838, 310.5284119, -411.0600281, 410.7662048
3: -105.4614182, 402.5131226, -106.1514359, 403.5927734, -509.0541992, 508.6645508
4: -90.0528412, 365.8195190, -90.6445084, 367.0319824, -457.0848389, 456.4640198

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7923764, upper bound: 339.7699983
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7923764, upper bound: 339.8024352
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -77.7812729, 263.4603271, -84.6927872, 284.6872253, -362.4684753, 348.1530762
1: -109.1529846, 261.0677795, -118.9581451, 282.3569946, -391.5099792, 380.0259399
2: -92.4850464, 287.5709229, -100.8533630, 310.9400940, -403.4250793, 388.4242554
3: -97.2033539, 374.3854675, -105.8310165, 404.3847656, -501.5881348, 480.2164307
4: -82.9996109, 340.1309814, -90.3477631, 367.4818726, -450.4814758, 430.4787292

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7891121, upper bound: 339.7880658
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7891121, upper bound: 339.7986506
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -84.3763123, 283.3912354, -85.4343643, 286.9955750, -371.3718872, 368.8255920
1: -118.5572128, 281.0944824, -120.0115356, 284.6641541, -403.2213745, 401.1059570
2: -100.5316162, 309.5450439, -101.7520218, 313.4716492, -414.0032654, 411.2970581
3: -105.4614182, 402.5131226, -106.7577209, 407.6237488, -513.0851440, 509.2707825
4: -90.0528412, 365.8195190, -91.1399841, 370.4416504, -460.4945068, 456.9595032

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7949463, upper bound: 339.7882300
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7949463, upper bound: 339.8011052
time: 1.07 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.81 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.81
Output dim: 0, lower bound: -339.7674618, upper bound: 339.7674618
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -339.7674618, upper bound: 339.7965074
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -339.7965074, upper bound: 339.7722590
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -339.7965074, upper bound: 339.8045731
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -339.7683776, upper bound: 339.7850231
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -339.7683776, upper bound: 339.7923764
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -339.7999904, upper bound: 339.7931278
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -339.7999904, upper bound: 339.8027842
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -339.7850230, upper bound: 339.7683776
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -339.7850230, upper bound: 339.7999904
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -339.7923764, upper bound: 339.7699983
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -339.7923764, upper bound: 339.8024352
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -339.7891121, upper bound: 339.7880658
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -339.7891121, upper bound: 339.7986506
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -339.7949463, upper bound: 339.7882300
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -339.7949463, upper bound: 339.8011052

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -76.5251465, 258.0331726, -84.2214584, 281.0291443, -357.5542908, 342.2546387
1: -107.2658615, 255.8545990, -118.1812286, 278.9996948, -386.2654724, 374.0358276
2: -90.9418640, 281.8870544, -100.2603912, 307.3133545, -398.2552185, 382.1474304
3: -95.5396729, 366.7406006, -105.1281967, 399.3886414, -494.9283142, 471.8688049
4: -81.6038895, 333.3200378, -89.7864609, 363.2832031, -444.8870850, 423.1064758

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7631526, upper bound: 339.7530758
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7557520, upper bound: 339.7519158
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -84.2214584, 281.0291443, -76.5251465, 258.0331726, -342.2546387, 357.5542908
1: -118.1812286, 278.9996948, -107.2658615, 255.8545990, -374.0358276, 386.2654724
2: -100.2603912, 307.3133545, -90.9418640, 281.8870544, -382.1474609, 398.2552185
3: -105.1281967, 399.3886414, -95.5396729, 366.7406006, -471.8688049, 494.9283142
4: -89.7864609, 363.2832031, -81.6038895, 333.3200378, -423.1064758, 444.8870850

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7960613, upper bound: 339.7715555
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7949675, upper bound: 339.7712641
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -84.2214584, 281.0291443, -84.2214584, 281.0291443, -365.2506104, 365.2506104
1: -118.1812286, 278.9996948, -118.1812286, 278.9996948, -397.1808777, 397.1808472
2: -100.2603912, 307.3133545, -100.2603912, 307.3133545, -407.5736694, 407.5736694
3: -105.1281967, 399.3886414, -105.1281967, 399.3886414, -504.5168457, 504.5168457
4: -89.7864609, 363.2832031, -89.7864609, 363.2832031, -453.0696716, 453.0696716

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7960613, upper bound: 339.7999548
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7949675, upper bound: 339.7991227
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -76.5251465, 258.0331726, -77.7812729, 263.4603271, -339.9854736, 335.8143921
1: -107.2658615, 255.8545990, -109.1529846, 261.0677795, -368.3335876, 365.0075684
2: -90.9418640, 281.8870544, -92.4850464, 287.5709229, -378.5127869, 374.3721008
3: -95.5396729, 366.7406006, -97.2033539, 374.3854675, -469.9251099, 463.9439697
4: -81.6038895, 333.3200378, -82.9996109, 340.1309814, -421.7348633, 416.3196411

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7579234, upper bound: 339.7722367
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7537425, upper bound: 339.7724469
time: 1.34 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -76.5251465, 258.0331726, -84.3763123, 283.3912354, -359.9163818, 342.4094849
1: -107.2658615, 255.8545990, -118.5572128, 281.0944824, -388.3603210, 374.4118042
2: -90.9418640, 281.8870544, -100.5316162, 309.5450439, -400.4869080, 382.4186707
3: -95.5396729, 366.7406006, -105.4614182, 402.5131226, -498.0527649, 472.2020264
4: -81.6038895, 333.3200378, -90.0528412, 365.8195190, -447.4234009, 423.3728638

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7579234, upper bound: 339.7829169
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7537425, upper bound: 339.7827424
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -84.2214584, 281.0291443, -77.7812729, 263.4603271, -347.6817627, 358.8104248
1: -118.1812286, 278.9996948, -109.1529846, 261.0677795, -379.2489624, 388.1526489
2: -100.2603912, 307.3133545, -92.4850464, 287.5709229, -387.8312683, 399.7983704
3: -105.1281967, 399.3886414, -97.2033539, 374.3854675, -479.5136414, 496.5919800
4: -89.7864609, 363.2832031, -82.9996109, 340.1309814, -429.9174500, 446.2828064

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7985180, upper bound: 339.7900411
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7957283, upper bound: 339.7895355
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -84.2214584, 281.0291443, -84.3763123, 283.3912354, -367.6127014, 365.4054565
1: -118.1812286, 278.9996948, -118.5572128, 281.0944824, -399.2756653, 397.5569153
2: -100.2603912, 307.3133545, -100.5316162, 309.5450439, -409.8053589, 407.8449707
3: -105.1281967, 399.3886414, -105.4614182, 402.5131226, -507.6412964, 504.8500366
4: -89.7864609, 363.2832031, -90.0528412, 365.8195190, -455.6059570, 453.3360596

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7985180, upper bound: 339.7975333
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7957283, upper bound: 339.7969842
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -77.7812729, 263.4603271, -76.5251465, 258.0331726, -335.8143921, 339.9854431
1: -109.1529846, 261.0677795, -107.2658615, 255.8545990, -365.0075684, 368.3335876
2: -92.4850464, 287.5709229, -90.9418640, 281.8870544, -374.3721008, 378.5127869
3: -97.2033539, 374.3854675, -95.5396729, 366.7406006, -463.9439697, 469.9251404
4: -82.9996109, 340.1309814, -81.6038895, 333.3200378, -416.3196411, 421.7348633

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7848989, upper bound: 339.7670722
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7848989, upper bound: 339.7679521
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -77.7812729, 263.4603271, -84.2214584, 281.0291443, -358.8103943, 347.6817627
1: -109.1529846, 261.0677795, -118.1812286, 278.9996948, -388.1526489, 379.2489624
2: -92.4850464, 287.5709229, -100.2603912, 307.3133545, -399.7983704, 387.8312988
3: -97.2033539, 374.3854675, -105.1281967, 399.3886414, -496.5919800, 479.5136719
4: -82.9996109, 340.1309814, -89.7864609, 363.2832031, -446.2828064, 429.9174500

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7848989, upper bound: 339.7953711
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7848989, upper bound: 339.7957283
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -84.3763123, 283.3912354, -76.5251465, 258.0331726, -342.4094849, 359.9163818
1: -118.5572128, 281.0944824, -107.2658615, 255.8545990, -374.4118042, 388.3603210
2: -100.5316162, 309.5450439, -90.9418640, 281.8870544, -382.4186707, 400.4869080
3: -105.4614182, 402.5131226, -95.5396729, 366.7406006, -472.2020264, 498.0527344
4: -90.0528412, 365.8195190, -81.6038895, 333.3200378, -423.3728638, 447.4234009

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7922398, upper bound: 339.7693746
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7920663, upper bound: 339.7695057
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -84.3763123, 283.3912354, -84.2214584, 281.0291443, -365.4054565, 367.6127014
1: -118.5572128, 281.0944824, -118.1812286, 278.9996948, -397.5569153, 399.2756958
2: -100.5316162, 309.5450439, -100.2603912, 307.3133545, -407.8449707, 409.8053894
3: -105.4614182, 402.5131226, -105.1281967, 399.3886414, -504.8500366, 507.6412659
4: -90.0528412, 365.8195190, -89.7864609, 363.2832031, -453.3360596, 455.6059570

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7922398, upper bound: 339.7977143
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7920663, upper bound: 339.7954062
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -77.7812729, 263.4603271, -77.7812729, 263.4603271, -341.2415161, 341.2414551
1: -109.1529846, 261.0677795, -109.1529846, 261.0677795, -370.2207336, 370.2207336
2: -92.4850464, 287.5709229, -92.4850464, 287.5709229, -380.0559387, 380.0559387
3: -97.2033539, 374.3854675, -97.2033539, 374.3854675, -471.5888062, 471.5888062
4: -82.9996109, 340.1309814, -82.9996109, 340.1309814, -423.1305847, 423.1305847

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7886305, upper bound: 339.7858700
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7862273, upper bound: 339.7859224
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -77.7812729, 263.4603271, -84.3763123, 283.3912354, -361.1725159, 347.8366394
1: -109.1529846, 261.0677795, -118.5572128, 281.0944824, -390.2474365, 379.6250000
2: -92.4850464, 287.5709229, -100.5316162, 309.5450439, -402.0300598, 388.1025391
3: -97.2033539, 374.3854675, -105.4614182, 402.5131226, -499.7164612, 479.8468933
4: -82.9996109, 340.1309814, -90.0528412, 365.8195190, -448.8191223, 430.1838379

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7886305, upper bound: 339.7932790
time: 1.27 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7862273, upper bound: 339.7936353
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -84.3763123, 283.3912354, -77.7812729, 263.4603271, -347.8366089, 361.1725159
1: -118.5572128, 281.0944824, -109.1529846, 261.0677795, -379.6250000, 390.2474365
2: -100.5316162, 309.5450439, -92.4850464, 287.5709229, -388.1025391, 402.0300598
3: -105.4614182, 402.5131226, -97.2033539, 374.3854675, -479.8468933, 499.7164612
4: -90.0528412, 365.8195190, -82.9996109, 340.1309814, -430.1838379, 448.8191223

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7943020, upper bound: 339.7867514
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7927669, upper bound: 339.7859926
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -84.3763123, 283.3912354, -84.3763123, 283.3912354, -367.7675476, 367.7675476
1: -118.5572128, 281.0944824, -118.5572128, 281.0944824, -399.6517029, 399.6517029
2: -100.5316162, 309.5450439, -100.5316162, 309.5450439, -410.0766602, 410.0766602
3: -105.4614182, 402.5131226, -105.4614182, 402.5131226, -507.9745178, 507.9745178
4: -90.0528412, 365.8195190, -90.0528412, 365.8195190, -455.8723755, 455.8723755

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7943020, upper bound: 339.7957954
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7927669, upper bound: 339.7954013
time: 1.31 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.90 seconds
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7631526, upper bound: 339.7530758
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7557520, upper bound: 339.7519158
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7960613, upper bound: 339.7715555
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7949675, upper bound: 339.7712641
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7960613, upper bound: 339.7999548
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7949675, upper bound: 339.7991227
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7579234, upper bound: 339.7722367
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7537425, upper bound: 339.7724469
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7579234, upper bound: 339.7829169
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7537425, upper bound: 339.7827424
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7985180, upper bound: 339.7900411
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7957283, upper bound: 339.7895355
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7985180, upper bound: 339.7975333
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7957283, upper bound: 339.7969842
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7848989, upper bound: 339.7670722
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7848989, upper bound: 339.7679521
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7848989, upper bound: 339.7953711
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7848989, upper bound: 339.7957283
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7922398, upper bound: 339.7693746
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7920663, upper bound: 339.7695057
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7922398, upper bound: 339.7977143
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7920663, upper bound: 339.7954062
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7886305, upper bound: 339.7858700
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7862273, upper bound: 339.7859224
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7886305, upper bound: 339.7932790
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7862273, upper bound: 339.7936353
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7943020, upper bound: 339.7867514
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7927669, upper bound: 339.7859926
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7943020, upper bound: 339.7957954
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.90
Output dim: 0, lower bound: -339.7927669, upper bound: 339.7954013

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -81.0358963, 269.8756104, -75.8349380, 255.6295471, -336.6654358, 345.7105408
1: -113.6971054, 267.9498291, -106.3010788, 253.4710999, -367.1682129, 374.2509155
2: -96.4887772, 295.1441040, -90.1286392, 279.2605591, -375.7493286, 385.2727051
3: -101.1333313, 383.3225708, -94.6790390, 363.2782593, -464.4115906, 478.0016174
4: -86.4139786, 348.7895813, -80.8745956, 330.2017517, -416.6156921, 429.6641846

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7851273, upper bound: 339.7598175
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7856281, upper bound: 339.7561644
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -88.0888672, 292.2174377, -75.5493164, 254.6386719, -342.7275391, 367.7667236
1: -123.6191406, 290.1853027, -105.8704910, 252.4886169, -376.1077271, 396.0557556
2: -104.8960876, 319.7673645, -89.7613297, 278.2082520, -383.1042786, 409.5286865
3: -109.8945084, 414.7697754, -94.2998962, 361.9012451, -471.7957458, 509.0696716
4: -93.7987595, 377.7793884, -80.5506210, 328.9654541, -422.7642212, 458.3299866

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7528858, upper bound: 339.7631526
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7519158, upper bound: 339.7557520
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -81.0358963, 269.8756104, -83.5718384, 278.7544861, -359.7903748, 353.4474487
1: -113.6971054, 267.9498291, -117.2703323, 276.7453918, -390.4425049, 385.2201538
2: -96.4887772, 295.1441040, -99.4944305, 304.8318481, -401.3206177, 394.6385498
3: -101.1333313, 383.3225708, -104.3165131, 396.1117859, -497.2451172, 487.6390686
4: -86.4139786, 348.7895813, -89.1018066, 360.3281555, -446.7420654, 437.8913574

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7926490, upper bound: 339.7982819
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7924816, upper bound: 339.7913693
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -88.0888672, 292.2174377, -83.1301346, 277.1951599, -365.2839966, 375.3475647
1: -123.6191406, 290.1853027, -116.6237946, 275.2098694, -398.8290100, 406.8090820
2: -104.8960876, 319.7673645, -98.9470596, 303.1746216, -408.0706482, 418.7143555
3: -109.8945084, 414.7697754, -103.7441864, 393.9298706, -503.8243713, 518.5138550
4: -93.7987595, 377.7793884, -88.6152115, 358.3766174, -452.1753845, 466.3945923

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7908737, upper bound: 339.7982572
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7904185, upper bound: 339.7907907
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -68.7997971, 232.7159271, -77.3249512, 262.1282349, -330.9280396, 310.0408936
1: -96.5648880, 230.8061523, -108.5776291, 259.7243347, -356.2892151, 339.3837891
2: -81.8181915, 254.3222504, -91.9891891, 286.0905151, -367.9086914, 346.3114319
3: -86.0115204, 330.8748779, -96.6854477, 372.4770203, -458.4885254, 427.5603333
4: -73.4907990, 300.7060547, -82.5522842, 338.3921204, -411.8829346, 383.2583313

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7565738, upper bound: 339.7721936
time: 1.49 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7579103, upper bound: 339.7717855
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -72.0914078, 243.8674316, -76.8968582, 260.6189575, -332.7103577, 320.7642822
1: -100.8317337, 241.7197113, -107.8650055, 258.2361755, -359.0679016, 349.5847168
2: -85.4771194, 266.3782654, -91.3917313, 284.4545288, -369.9316406, 357.7699890
3: -89.8558121, 346.8189087, -96.0663757, 370.3821716, -460.2379761, 442.8852844
4: -76.8516617, 315.0495605, -82.0360641, 336.4607849, -413.3123779, 397.0856323

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7524458, upper bound: 339.7723840
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7537406, upper bound: 339.7710993
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -68.7997971, 232.7159271, -83.9804077, 282.1568298, -350.9566345, 316.6963501
1: -96.5648880, 230.8061523, -118.0245209, 279.8540039, -376.4188843, 348.8306580
2: -81.8181915, 254.3222504, -100.0777130, 308.1833801, -390.0015869, 354.3999329
3: -86.0115204, 330.8748779, -104.9847488, 400.7438049, -486.7553101, 435.8596191
4: -73.4907990, 300.7060547, -89.6458054, 364.2147522, -437.7055359, 390.3518677

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7593637, upper bound: 339.7828943
time: 1.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7593080, upper bound: 339.7818886
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -72.0914078, 243.8674316, -83.5371475, 280.7799683, -352.8713684, 327.4045715
1: -100.8317337, 241.7197113, -117.3672256, 278.4933472, -379.3250427, 359.0869446
2: -85.4771194, 266.3782654, -99.5176392, 306.6687012, -392.1457520, 365.8958435
3: -89.8558121, 346.8189087, -104.4086990, 398.8486633, -488.7044678, 451.2276001
4: -76.8516617, 315.0495605, -89.1505814, 362.4498596, -439.3014526, 404.2001343

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7551507, upper bound: 339.7827181
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7550062, upper bound: 339.7811055
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -81.0358963, 269.8756104, -77.0467682, 260.9003601, -341.9362488, 346.9223328
1: -113.6971054, 267.9498291, -108.1254807, 258.5303040, -372.2273865, 376.0753174
2: -96.4887772, 295.1441040, -91.6155853, 284.7749939, -381.2637329, 386.7596436
3: -101.1333313, 383.3225708, -96.2873993, 370.7064514, -471.8397522, 479.6099243
4: -86.4139786, 348.7895813, -82.2205124, 336.8143311, -423.2282410, 431.0101013

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7868629, upper bound: 339.7829761
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7864414, upper bound: 339.7766313
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -88.0888672, 292.2174377, -76.8286438, 260.1546021, -348.2434692, 369.0460815
1: -123.6191406, 290.1853027, -107.7899933, 257.7904968, -381.4096375, 397.9752808
2: -104.8960876, 319.7673645, -91.3312607, 283.9741821, -388.8702087, 411.0986328
3: -109.8945084, 414.7697754, -95.9923172, 369.6796265, -479.5741272, 510.7620850
4: -93.7987595, 377.7793884, -81.9714279, 335.8953552, -429.6940918, 459.7508240

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7861814, upper bound: 339.7849009
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7852857, upper bound: 339.7762803
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -81.0358963, 269.8756104, -83.7250900, 281.1330872, -362.1689758, 353.6007080
1: -113.6971054, 267.9498291, -117.6471176, 278.8516541, -392.5487671, 385.5969238
2: -96.4887772, 295.1441040, -99.7633514, 307.0733337, -403.5621033, 394.9074707
3: -101.1333313, 383.3225708, -104.6505585, 399.2602234, -500.3935547, 487.9731140
4: -86.4139786, 348.7895813, -89.3670731, 362.8866577, -449.3005371, 438.1566162

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7898378, upper bound: 339.7928931
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7889062, upper bound: 339.7869580
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -88.0888672, 292.2174377, -83.2668457, 279.5134583, -367.6023254, 375.4842529
1: -123.6191406, 290.1853027, -116.9737854, 277.2566223, -400.8757629, 407.1590881
2: -104.8960876, 319.7673645, -99.1941757, 305.3367004, -410.2327271, 418.9614868
3: -109.8945084, 414.7697754, -104.0540390, 396.9930725, -506.8875732, 518.8237305
4: -93.7987595, 377.7793884, -88.8605347, 360.8471375, -454.6459045, 466.6399231

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7887099, upper bound: 339.7942436
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7871808, upper bound: 339.7863784
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -74.5017242, 252.0583954, -75.8349380, 255.6295471, -330.1312866, 327.8933411
1: -104.5777283, 249.7615204, -106.3010788, 253.4710999, -358.0488281, 356.0625916
2: -88.6153183, 275.1112366, -90.1286392, 279.2605591, -367.8758545, 365.2398376
3: -93.1252823, 358.0035706, -94.6790390, 363.2782593, -456.4035339, 452.6826172
4: -79.5323639, 325.3593140, -80.8745956, 330.2017517, -409.7341309, 406.2339172

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7721936, upper bound: 339.7565738
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7723840, upper bound: 339.7524458
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -82.1193466, 276.7013550, -75.5493164, 254.6386719, -336.7579956, 352.2506409
1: -115.4125595, 274.2441406, -105.8704910, 252.4886169, -367.9010925, 380.1146240
2: -97.7877274, 302.1891785, -89.7613297, 278.2082520, -375.9959412, 391.9505005
3: -102.6778336, 392.8051453, -94.2998962, 361.9012451, -464.5790710, 487.1050415
4: -87.5935593, 357.3958130, -80.5506210, 328.9654541, -416.5589905, 437.9464111

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7594418, upper bound: 339.7637479
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7584719, upper bound: 339.7563473
time: 1.62 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -74.5017242, 252.0583954, -83.5718384, 278.7544861, -353.2562256, 335.6302490
1: -104.5777283, 249.7615204, -117.2703323, 276.7453918, -381.3230896, 367.0318604
2: -88.6153183, 275.1112366, -99.4944305, 304.8318481, -393.4471130, 374.6056519
3: -93.1252823, 358.0035706, -104.3165131, 396.1117859, -489.2370605, 462.3200684
4: -79.5323639, 325.3593140, -89.1018066, 360.3281555, -439.8605347, 414.4611206

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7775057, upper bound: 339.7921236
time: 1.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7775562, upper bound: 339.7846673
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -82.1193466, 276.7013550, -83.1301346, 277.1951599, -359.3144531, 359.8314819
1: -115.4125595, 274.2441406, -116.6237946, 275.2098694, -390.6224365, 390.8679199
2: -97.7877274, 302.1891785, -98.9470596, 303.1746216, -400.9623108, 401.1361389
3: -102.6778336, 392.8051453, -103.7441864, 393.9298706, -496.6076965, 496.5493164
4: -87.5935593, 357.3958130, -88.6152115, 358.3766174, -445.9701233, 446.0110168

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7764023, upper bound: 339.7929903
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7762803, upper bound: 339.7852857
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -81.1669083, 272.2276917, -75.8349380, 255.6295471, -336.7963867, 348.0626221
1: -114.0635605, 270.0139465, -106.3010788, 253.4710999, -367.5346680, 376.3150330
2: -96.7370834, 297.3360596, -90.1286392, 279.2605591, -375.9975891, 387.4646912
3: -101.4570694, 386.4553833, -94.6790390, 363.2782593, -464.7353210, 481.1344299
4: -86.6637650, 351.3312073, -80.8745956, 330.2017517, -416.8655090, 432.2058105

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7828943, upper bound: 339.7593637
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7827181, upper bound: 339.7551507
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -88.2225113, 294.4195557, -75.5493164, 254.6386719, -342.8611755, 369.9688416
1: -123.9681091, 292.1536255, -105.8704910, 252.4886169, -376.4566345, 398.0241089
2: -105.1256409, 321.8575439, -89.7613297, 278.2082520, -383.3338928, 411.6188660
3: -110.1987991, 417.7736511, -94.2998962, 361.9012451, -472.1000366, 512.0735474
4: -94.0284424, 380.2252808, -80.5506210, 328.9654541, -422.9938965, 460.7759094

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7818886, upper bound: 339.7593080
time: 1.27 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7811055, upper bound: 339.7550062
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -81.1669083, 272.2276917, -83.5718384, 278.7544861, -359.9213257, 355.7995300
1: -114.0635605, 270.0139465, -117.2703323, 276.7453918, -390.8089294, 387.2842712
2: -96.7370834, 297.3360596, -99.4944305, 304.8318481, -401.5688782, 396.8305054
3: -101.4570694, 386.4553833, -104.3165131, 396.1117859, -497.5688477, 490.7719116
4: -86.6637650, 351.3312073, -89.1018066, 360.3281555, -446.9919128, 440.4330139

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7877915, upper bound: 339.7948815
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7867101, upper bound: 339.7865169
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -88.2225113, 294.4195557, -83.1301346, 277.1951599, -365.4176331, 377.5496826
1: -123.9681091, 292.1536255, -116.6237946, 275.2098694, -399.1779480, 408.7774048
2: -105.1256409, 321.8575439, -98.9470596, 303.1746216, -408.3002625, 420.8045349
3: -110.1987991, 417.7736511, -103.7441864, 393.9298706, -504.1286621, 521.5177002
4: -94.0284424, 380.2252808, -88.6152115, 358.3766174, -452.4050598, 468.8404846

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7851631, upper bound: 339.7932032
time: 1.36 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7836976, upper bound: 339.7828134
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -74.5017242, 252.0583954, -77.0467682, 260.9003601, -335.4020996, 329.1051636
1: -104.5777283, 249.7615204, -108.1254807, 258.5303040, -363.1079712, 357.8869629
2: -88.6153183, 275.1112366, -91.6155853, 284.7749939, -373.3902893, 366.7267761
3: -93.1252823, 358.0035706, -96.2873993, 370.7064514, -463.8317261, 454.2909241
4: -79.5323639, 325.3593140, -82.2205124, 336.8143311, -416.3466797, 407.5798340

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7747778, upper bound: 339.7806888
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7734630, upper bound: 339.7701927
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -82.1193466, 276.7013550, -76.8286438, 260.1546021, -342.2739563, 353.5299988
1: -115.4125595, 274.2441406, -107.7899933, 257.7904968, -373.2030640, 382.0341187
2: -97.7877274, 302.1891785, -91.3312607, 283.9741821, -381.7618713, 393.5204163
3: -102.6778336, 392.8051453, -95.9923172, 369.6796265, -472.3574524, 488.7974548
4: -87.5935593, 357.3958130, -81.9714279, 335.8953552, -423.4888916, 439.3672485

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727678, upper bound: 339.7810283
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7710921, upper bound: 339.7702447
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -74.5017242, 252.0583954, -83.7250900, 281.1330872, -355.6348267, 335.7834778
1: -104.5777283, 249.7615204, -117.6471176, 278.8516541, -383.4293823, 367.4085999
2: -88.6153183, 275.1112366, -99.7633514, 307.0733337, -395.6886292, 374.8745728
3: -93.1252823, 358.0035706, -104.6505585, 399.2602234, -492.3854980, 462.6541138
4: -79.5323639, 325.3593140, -89.3670731, 362.8866577, -442.4190063, 414.7263489

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7764239, upper bound: 339.7895750
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7754224, upper bound: 339.7802665
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -82.1193466, 276.7013550, -83.2668457, 279.5134583, -361.6328125, 359.9681702
1: -115.4125595, 274.2441406, -116.9737854, 277.2566223, -392.6691895, 391.2179260
2: -97.7877274, 302.1891785, -99.1941757, 305.3367004, -403.1243896, 401.3832703
3: -102.6778336, 392.8051453, -104.0540390, 396.9930725, -499.6708984, 496.8591919
4: -87.5935593, 357.3958130, -88.8605347, 360.8471375, -448.4407043, 446.2563477

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7746307, upper bound: 339.7899761
time: 1.41 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727062, upper bound: 339.7811054
time: 1.31 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -81.1669083, 272.2276917, -77.0467682, 260.9003601, -342.0671997, 349.2743835
1: -114.0635605, 270.0139465, -108.1254807, 258.5303040, -372.5938110, 378.1394348
2: -96.7370834, 297.3360596, -91.6155853, 284.7749939, -381.5119934, 388.9516296
3: -101.4570694, 386.4553833, -96.2873993, 370.7064514, -472.1635132, 482.7427368
4: -86.6637650, 351.3312073, -82.2205124, 336.8143311, -423.4780884, 433.5517273

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7828943, upper bound: 339.7816805
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7827181, upper bound: 339.7707040
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -88.2225113, 294.4195557, -76.8286438, 260.1546021, -348.3771057, 371.2481995
1: -123.9681091, 292.1536255, -107.7899933, 257.7904968, -381.7585754, 399.9436035
2: -105.1256409, 321.8575439, -91.3312607, 283.9741821, -389.0998230, 413.1888123
3: -110.1987991, 417.7736511, -95.9923172, 369.6796265, -479.8784180, 513.7659912
4: -94.0284424, 380.2252808, -81.9714279, 335.8953552, -429.9237976, 462.1967163

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7818886, upper bound: 339.7814900
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7811023, upper bound: 339.7703141
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -81.1669083, 272.2276917, -83.7250900, 281.1330872, -362.2999573, 355.9527893
1: -114.0635605, 270.0139465, -117.6471176, 278.8516541, -392.9152222, 387.6610718
2: -96.7370834, 297.3360596, -99.7633514, 307.0733337, -403.8103638, 397.0994263
3: -101.4570694, 386.4553833, -104.6505585, 399.2602234, -500.7172852, 491.1059265
4: -86.6637650, 351.3312073, -89.3670731, 362.8866577, -449.5504150, 440.6982727

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7863229, upper bound: 339.7922360
time: 1.32 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7848311, upper bound: 339.7829758
time: 1.20 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -88.2225113, 294.4195557, -83.2668457, 279.5134583, -367.7359619, 377.6863708
1: -123.9681091, 292.1536255, -116.9737854, 277.2566223, -401.2247009, 409.1274109
2: -105.1256409, 321.8575439, -99.1941757, 305.3367004, -410.4623413, 421.0516663
3: -110.1987991, 417.7736511, -104.0540390, 396.9930725, -507.1918640, 521.8274536
4: -94.0284424, 380.2252808, -88.8605347, 360.8471375, -454.8755798, 469.0858154

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7846889, upper bound: 339.7921551
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7827782, upper bound: 339.7827801
time: 0.83 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.64 seconds
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7851273, upper bound: 339.7598175
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7856281, upper bound: 339.7561644
IS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7528858, upper bound: 339.7631526
IS_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7519158, upper bound: 339.7557520
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7926490, upper bound: 339.7982819
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7924816, upper bound: 339.7913693
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7908737, upper bound: 339.7982572
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7904185, upper bound: 339.7907907
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7565738, upper bound: 339.7721936
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7579103, upper bound: 339.7717855
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7524458, upper bound: 339.7723840
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7537406, upper bound: 339.7710993
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7593637, upper bound: 339.7828943
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7593080, upper bound: 339.7818886
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7551507, upper bound: 339.7827181
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7550062, upper bound: 339.7811055
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7868629, upper bound: 339.7829761
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7864414, upper bound: 339.7766313
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7861814, upper bound: 339.7849009
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7852857, upper bound: 339.7762803
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7898378, upper bound: 339.7928931
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7889062, upper bound: 339.7869580
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7887099, upper bound: 339.7942436
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7871808, upper bound: 339.7863784
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7721936, upper bound: 339.7565738
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7723840, upper bound: 339.7524458
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7594418, upper bound: 339.7637479
IS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7584719, upper bound: 339.7563473
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7775057, upper bound: 339.7921236
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7775562, upper bound: 339.7846673
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7764023, upper bound: 339.7929903
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7762803, upper bound: 339.7852857
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7828943, upper bound: 339.7593637
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7827181, upper bound: 339.7551507
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7818886, upper bound: 339.7593080
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7811055, upper bound: 339.7550062
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7877915, upper bound: 339.7948815
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7867101, upper bound: 339.7865169
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7851631, upper bound: 339.7932032
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7836976, upper bound: 339.7828134
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7747778, upper bound: 339.7806888
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7734630, upper bound: 339.7701927
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7727678, upper bound: 339.7810283
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7710921, upper bound: 339.7702447
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7764239, upper bound: 339.7895750
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7754224, upper bound: 339.7802665
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7746307, upper bound: 339.7899761
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7727062, upper bound: 339.7811054
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7828943, upper bound: 339.7816805
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7827181, upper bound: 339.7707040
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7818886, upper bound: 339.7814900
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7811023, upper bound: 339.7703141
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7863229, upper bound: 339.7922360
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7848311, upper bound: 339.7829758
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7846889, upper bound: 339.7921551
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 0, lower bound: -339.7827782, upper bound: 339.7827801

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -80.6251068, 268.5622864, -68.0644073, 230.1903229, -310.8154297, 336.6267090
1: -113.1321640, 266.6321106, -95.5367203, 228.2873230, -341.4194641, 362.1688232
2: -96.0067368, 293.7008362, -80.9524918, 251.5458832, -347.5526123, 374.6532593
3: -100.6277084, 381.4436035, -85.0938187, 327.2500916, -427.8777771, 466.5374146
4: -85.9815369, 347.0801086, -72.7157516, 297.4243469, -383.4058533, 419.7958679

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7851273, upper bound: 339.7561644
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7851273, upper bound: 339.7561644
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -80.1710358, 267.1965637, -71.2899094, 241.0487671, -321.2197876, 338.4864807
1: -112.4870682, 265.2853394, -99.7108459, 238.9394073, -351.4263916, 364.9961243
2: -95.4577026, 292.1950989, -84.5325928, 263.3031921, -358.7608948, 376.7276917
3: -100.0618591, 379.5616150, -88.8562698, 342.7657776, -442.8276367, 468.4178772
4: -85.4968414, 345.3378296, -76.0060349, 311.4018250, -396.8986816, 421.3438416

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7856281, upper bound: 339.7561644
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7856281, upper bound: 339.7561644
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -80.6251068, 268.5622864, -75.9036713, 253.4705200, -334.0956421, 344.4659424
1: -113.1321640, 266.6321106, -106.5353851, 251.7314301, -364.8635559, 373.1674805
2: -96.0067368, 293.7008362, -90.3486481, 277.3031921, -373.3099365, 384.0494080
3: -100.6277084, 381.4436035, -94.7607803, 360.2982178, -460.9259033, 476.2043762
4: -85.9815369, 347.0801086, -80.9585953, 327.7587891, -413.7402954, 428.0386963

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7924816, upper bound: 339.7913693
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7924816, upper bound: 339.7913693
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -80.1710358, 267.1965637, -79.0778885, 264.5686951, -344.7397461, 346.2744141
1: -112.4870682, 265.2853394, -110.8519669, 262.5954285, -375.0824890, 376.1372681
2: -95.4577026, 292.1950989, -94.0325089, 289.2453918, -384.7030945, 386.2275391
3: -100.0618591, 379.5616150, -98.6358490, 376.1281738, -476.1900330, 478.1974487
4: -85.4968414, 345.3378296, -84.2579956, 342.0183716, -427.5151978, 429.5958252

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7816955, upper bound: 339.7464337
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7855922, upper bound: 339.7841271
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7876587, upper bound: 339.7867959
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -87.6950302, 290.9225769, -75.6332703, 252.4937592, -340.1887512, 366.5558472
1: -123.0724335, 288.9018860, -106.1355591, 250.8007202, -373.8731689, 395.0374451
2: -104.4310379, 318.3555603, -90.0048447, 276.2969360, -380.7279053, 408.3603821
3: -109.4067307, 412.9260559, -94.4101868, 358.9679871, -468.3746948, 507.3362427
4: -93.3825607, 376.1003723, -80.6565399, 326.5873108, -419.9698792, 456.7568665

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7904185, upper bound: 339.7907907
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7904185, upper bound: 339.7907907
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -87.1124954, 289.2026672, -78.7869949, 263.5641785, -350.6766663, 367.9896545
1: -122.2636337, 287.1802979, -110.4110870, 261.6183167, -383.8819580, 397.5913391
2: -103.7386169, 316.4451294, -93.6563797, 288.2019653, -391.9405823, 410.1015015
3: -108.6913071, 410.5332336, -98.2467728, 374.7257690, -483.4170837, 508.7799988
4: -92.7662048, 373.8969421, -83.9327621, 340.7789612, -433.5451660, 457.8297119

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7813331, upper bound: 339.7478274
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7424259, upper bound: 339.7424259
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -68.0644073, 230.1903229, -74.0451126, 250.7230988, -318.7875061, 304.2354126
1: -95.5367203, 228.2873230, -104.0023956, 248.4122467, -343.9489746, 332.2897339
2: -80.9524918, 251.5458832, -88.1192245, 273.6270447, -354.5795288, 339.6651001
3: -85.0938187, 327.2500916, -92.6066742, 356.0893555, -441.1831665, 419.8567505
4: -72.7157516, 297.4243469, -79.0839767, 323.6178589, -396.3336182, 376.5083313

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7390123, upper bound: 339.7689860
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7565738, upper bound: 339.7721936
time: 1.27 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -67.7498703, 229.0433197, -81.7110138, 275.3930969, -343.1428833, 310.7543335
1: -95.0691910, 227.1925659, -114.8497009, 272.9419861, -368.0111084, 342.0422668
2: -80.5536346, 250.3590240, -97.3085480, 300.7554321, -381.3090210, 347.6675415
3: -84.6835709, 325.6418152, -102.1751022, 390.9418640, -475.6253662, 427.8169250
4: -72.3654099, 295.9992676, -87.1649323, 355.6936646, -428.0590820, 383.1641846

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7546485, upper bound: 339.7479521
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7396345, upper bound: 339.7461218
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -71.2899094, 241.0487671, -73.6408310, 249.2856293, -320.5755310, 314.6896057
1: -99.7108459, 238.9394073, -103.3216629, 247.0043640, -346.7151489, 342.2609558
2: -84.5325928, 263.3031921, -87.5480804, 272.0698547, -356.6024170, 350.8512573
3: -88.8562698, 342.7657776, -92.0163040, 354.1039734, -442.9602356, 434.7820740
4: -76.0060349, 311.4018250, -78.5888367, 321.7823792, -397.7883911, 389.9906616

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7462417, upper bound: 339.7714830
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7524458, upper bound: 339.7723840
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -71.2137833, 240.8298492, -81.2250595, 273.9473572, -345.1611328, 322.0548706
1: -99.5777740, 238.7317963, -114.1603241, 271.5029907, -371.0807495, 352.8920593
2: -84.4155197, 263.1067810, -96.7192917, 299.1594238, -383.5748901, 359.8260803
3: -88.7422943, 342.4947510, -101.5682297, 388.9327393, -477.6750488, 444.0629883
4: -75.9162445, 311.1607666, -86.6439209, 353.8410034, -429.7572327, 397.8046265

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7520238, upper bound: 339.7472435
time: 1.39 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7456043, upper bound: 339.7466458
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -68.0644073, 230.1903229, -80.7669601, 270.9548035, -339.0192261, 310.9572754
1: -95.5367203, 228.2873230, -113.5144272, 268.7375793, -364.2742920, 341.8017578
2: -80.9524918, 251.5458832, -96.2689514, 295.9350586, -376.8875122, 347.8148193
3: -85.0938187, 327.2500916, -100.9660950, 384.6330872, -469.7268982, 428.2161865
4: -72.7157516, 297.4243469, -86.2433472, 349.6759644, -422.3917236, 383.6676941

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7408153, upper bound: 339.7728719
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7585890, upper bound: 339.7770999
time: 1.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -67.7498703, 229.0433197, -87.8358994, 293.1613159, -360.9111633, 316.8791809
1: -95.0691910, 227.1925659, -123.4324188, 290.9053345, -385.9745178, 350.6250000
2: -80.5536346, 250.3590240, -104.6696396, 320.4808655, -401.0344543, 355.0286560
3: -84.6835709, 325.6418152, -109.7212601, 415.9805908, -500.6640930, 435.3630676
4: -72.3654099, 295.9992676, -93.6202164, 378.5888062, -450.9542236, 389.6194763

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7411460, upper bound: 339.7724679
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7589506, upper bound: 339.7766597
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -71.2899094, 241.0487671, -80.3281021, 269.6324768, -340.9223938, 321.3768616
1: -99.7108459, 238.9394073, -112.8804398, 267.4279480, -367.1387329, 351.8197632
2: -84.5325928, 263.3031921, -95.7285004, 294.4765320, -379.0091248, 359.0316772
3: -88.8562698, 342.7657776, -100.4107513, 382.8062439, -471.6625061, 443.1765137
4: -76.0060349, 311.4018250, -85.7668686, 347.9750061, -423.9810181, 397.1687012

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7480418, upper bound: 339.7751666
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7542398, upper bound: 339.7763183
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -71.2137833, 240.8298492, -87.2984085, 291.6027222, -362.8164978, 328.1282654
1: -99.5777740, 238.7317963, -122.6833878, 289.3379211, -388.9157104, 361.4151917
2: -84.4155197, 263.1067810, -104.0266800, 318.7463989, -403.1618958, 367.1334534
3: -88.7422943, 342.4947510, -109.0581818, 413.8095703, -502.5518799, 451.5528870
4: -75.9162445, 311.1607666, -93.0484009, 376.5835266, -452.4997559, 404.2091675

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7079937, upper bound: 339.7598812
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7550062, upper bound: 339.7811055
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -80.6251068, 268.5622864, -69.1004791, 234.8384399, -315.4635010, 337.6627808
1: -113.1321640, 266.6321106, -97.1163101, 232.7198792, -345.8520508, 363.7484131
2: -96.0067368, 293.7008362, -82.2335587, 256.3724060, -352.3791504, 375.9343872
3: -100.6277084, 381.4436035, -86.4816818, 333.8312988, -434.4589844, 467.9252930
4: -85.9815369, 347.0801086, -73.8685989, 303.2889099, -389.2704163, 420.9487000

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7864414, upper bound: 339.7766313
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7864414, upper bound: 339.7766313
time: 1.17 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -80.1710358, 267.1965637, -72.7415161, 247.1914825, -327.3625183, 339.9380798
1: -112.4870682, 265.2853394, -101.8624344, 244.8570862, -357.3441467, 367.1477356
2: -95.4577026, 292.1950989, -86.3083038, 269.7597046, -365.2174072, 378.5033264
3: -100.0618591, 379.5616150, -90.7595215, 351.3945007, -451.4563293, 470.3211365
4: -85.4968414, 345.3378296, -77.6182098, 319.1378784, -404.6347046, 422.9560547

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7864414, upper bound: 339.7766313
time: 1.55 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7864414, upper bound: 339.7766313
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -87.6950302, 290.9225769, -68.8415070, 233.9081268, -321.6031494, 359.7639771
1: -123.0724335, 288.9018860, -96.7290573, 231.8316498, -354.9040833, 385.6309509
2: -104.4310379, 318.3555603, -81.9028854, 255.4130859, -359.8441162, 400.2584534
3: -109.4067307, 412.9260559, -86.1418457, 332.5274048, -441.9341125, 499.0679016
4: -93.3825607, 376.1003723, -73.5806732, 302.1335449, -395.5161133, 449.6810303

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7852857, upper bound: 339.7762803
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7852857, upper bound: 339.7762803
time: 1.30 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -87.1124954, 289.2026672, -72.6733322, 247.0215912, -334.1340637, 361.8760071
1: -122.2636337, 287.1802979, -101.7410355, 244.6726990, -366.9363403, 388.9213257
2: -103.7386169, 316.4451294, -86.2024460, 269.5836792, -373.3222961, 402.6475830
3: -108.6913071, 410.5332336, -90.6556244, 351.1803589, -459.8716736, 501.1888428
4: -92.7662048, 373.8969421, -77.5387955, 318.9519348, -411.7181396, 451.4357300

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7808463, upper bound: 339.7514072
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7427929, upper bound: 339.7463856
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -80.6251068, 268.5622864, -75.9359436, 255.4910583, -336.1161499, 344.4982300
1: -113.1321640, 266.6321106, -106.7664108, 253.4515839, -366.5837402, 373.3984680
2: -96.0067368, 293.7008362, -90.4846191, 279.1177979, -375.1245422, 384.1853638
3: -100.6277084, 381.4436035, -94.9580994, 362.9506836, -463.5783691, 476.4016724
4: -85.9815369, 347.0801086, -81.0959396, 329.8342896, -415.8157959, 428.1760559

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7889062, upper bound: 339.7869580
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7889062, upper bound: 339.7869580
time: 1.36 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -80.1710358, 267.1965637, -79.3890152, 267.4740601, -347.6450806, 346.5855713
1: -112.4870682, 265.2853394, -111.4268417, 265.2202454, -377.7072754, 376.7121887
2: -95.4577026, 292.1950989, -94.4714813, 292.0548401, -387.5125427, 386.6665039
3: -100.0618591, 379.5616150, -99.1526260, 380.0376587, -480.0995178, 478.7142334
4: -85.4968414, 345.3378296, -84.6935501, 345.2489014, -430.7457275, 430.0313721

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7889062, upper bound: 339.7869580
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7889062, upper bound: 339.7869580
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -87.6950302, 290.9225769, -75.6705246, 254.5117798, -342.2068176, 366.5930481
1: -123.0724335, 288.9018860, -106.3717117, 252.5263062, -375.5987549, 395.2735901
2: -104.4310379, 318.3555603, -90.1485367, 278.1208801, -382.5518799, 408.5040894
3: -109.4067307, 412.9260559, -94.6121979, 361.5787964, -470.9855347, 507.5382690
4: -93.3825607, 376.1003723, -80.7989960, 328.6356201, -422.0181885, 456.8993530

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7871808, upper bound: 339.7863784
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7871808, upper bound: 339.7863784
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -87.1124954, 289.2026672, -79.1141891, 266.5035400, -353.6160278, 368.3168640
1: -122.2636337, 287.1802979, -111.0049973, 264.2621155, -386.5257568, 398.1853027
2: -103.7386169, 316.4451294, -94.1135712, 291.0484619, -394.7870789, 410.5586853
3: -108.6913071, 410.5332336, -98.7804718, 378.6914368, -487.3827515, 509.3136902
4: -92.7662048, 373.8969421, -84.3830643, 344.0525513, -436.8187561, 458.2799988

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7871808, upper bound: 339.7863784
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7871808, upper bound: 339.7863784
time: 0.97 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.39 seconds
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7851273, upper bound: 339.7561644
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7851273, upper bound: 339.7561644
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7856281, upper bound: 339.7561644
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7856281, upper bound: 339.7561644
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7924816, upper bound: 339.7913693
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7924816, upper bound: 339.7913693
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7855922, upper bound: 339.7841271
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7876587, upper bound: 339.7867959
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7904185, upper bound: 339.7907907
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7904185, upper bound: 339.7907907
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7813331, upper bound: 339.7478274
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7424259, upper bound: 339.7424259
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7390123, upper bound: 339.7689860
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7565738, upper bound: 339.7721936
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7546485, upper bound: 339.7479521
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7396345, upper bound: 339.7461218
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7462417, upper bound: 339.7714830
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7524458, upper bound: 339.7723840
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7520238, upper bound: 339.7472435
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7456043, upper bound: 339.7466458
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7408153, upper bound: 339.7728719
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7585890, upper bound: 339.7770999
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7411460, upper bound: 339.7724679
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7589506, upper bound: 339.7766597
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7480418, upper bound: 339.7751666
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7542398, upper bound: 339.7763183
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7079937, upper bound: 339.7598812
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7550062, upper bound: 339.7811055
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7864414, upper bound: 339.7766313
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7864414, upper bound: 339.7766313
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7864414, upper bound: 339.7766313
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7864414, upper bound: 339.7766313
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7852857, upper bound: 339.7762803
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7852857, upper bound: 339.7762803
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7808463, upper bound: 339.7514072
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7427929, upper bound: 339.7463856
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7889062, upper bound: 339.7869580
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7889062, upper bound: 339.7869580
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7889062, upper bound: 339.7869580
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7889062, upper bound: 339.7869580
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7871808, upper bound: 339.7863784
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7871808, upper bound: 339.7863784
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7871808, upper bound: 339.7863784
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.39
Output dim: 0, lower bound: -339.7871808, upper bound: 339.7863784
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7721936, upper bound: 339.7565738
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7723840, upper bound: 339.7524458
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7775057, upper bound: 339.7921236
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7775562, upper bound: 339.7846673
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7764023, upper bound: 339.7929903
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7762803, upper bound: 339.7852857
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7828943, upper bound: 339.7593637
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7827181, upper bound: 339.7551507
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7818886, upper bound: 339.7593080
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7811055, upper bound: 339.7550062
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7877915, upper bound: 339.7948815
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7867101, upper bound: 339.7865169
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7851631, upper bound: 339.7932032
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7836976, upper bound: 339.7828134
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7747778, upper bound: 339.7806888
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7734630, upper bound: 339.7701927
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7727678, upper bound: 339.7810283
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7710921, upper bound: 339.7702447
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7764239, upper bound: 339.7895750
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7754224, upper bound: 339.7802665
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7746307, upper bound: 339.7899761
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7727062, upper bound: 339.7811054
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7828943, upper bound: 339.7816805
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7827181, upper bound: 339.7707040
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7818886, upper bound: 339.7814900
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7811023, upper bound: 339.7703141
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7863229, upper bound: 339.7922360
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7848311, upper bound: 339.7829758
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7846889, upper bound: 339.7921551
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -339.7827782, upper bound: 339.7827801
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=385.80084228515625
rel_dist={0: [-339.8055350744037, 339.8055350744037]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8041665, upper bound: 339.8048924
time: 1.19 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8041305, upper bound: 339.8041305
time: 1.15 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.55 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.55
Output dim: 0, lower bound: -339.8041665, upper bound: 339.8048924
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.55
Output dim: 0, lower bound: -339.8041305, upper bound: 339.8041305

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -87.7737961, 293.9982300, -379.0931702, 371.9442749
1: -119.3920059, 282.0884705, -123.1662445, 291.7587585, -411.1507263, 405.2546997
2: -101.2751236, 310.7092590, -104.4636459, 321.3109741, -422.5860596, 415.1728821
3: -106.2089005, 403.8296509, -109.5737915, 417.5669556, -523.7758789, 513.4033813
4: -90.6926956, 367.2429504, -93.5384369, 379.5859985, -470.2786865, 460.7813721

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7998110, upper bound: 339.7917297
time: 0.98 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8023114, upper bound: 339.8020642
time: 1.15 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -85.5067825, 287.2420959, -87.7549973, 294.1903076, -379.6970825, 374.9970398
1: -120.1114273, 284.9082947, -123.1697083, 291.8923950, -412.0038147, 408.0780029
2: -101.8361511, 313.7402954, -104.4544525, 321.4425049, -423.2786560, 418.1947327
3: -106.8467102, 407.9730225, -109.5743942, 417.8212891, -524.6679688, 517.5474243
4: -91.2148666, 370.7578735, -93.5370636, 379.7834473, -470.9983215, 464.2949219

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7983399, upper bound: 339.7911649
time: 1.07 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8010463, upper bound: 339.8010463
time: 1.04 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.60 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 0, lower bound: -339.7998110, upper bound: 339.7917297
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 0, lower bound: -339.8023114, upper bound: 339.8020642
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 0, lower bound: -339.7983399, upper bound: 339.7911649
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 0, lower bound: -339.8010463, upper bound: 339.8010463

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -81.4486237, 272.7717590, -79.6675186, 269.1990051, -350.6476440, 352.4392090
1: -114.2278366, 270.7118225, -111.7141953, 266.8656311, -381.0934753, 382.4260254
2: -96.8694229, 298.2447815, -94.6798553, 293.9727478, -390.8421326, 392.9246216
3: -101.6557465, 387.8109741, -99.4947586, 382.5419617, -484.1976929, 487.3057251
4: -86.8072586, 352.6760254, -84.9463425, 347.6036072, -434.4108582, 437.6222839

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7980131, upper bound: 339.7892662
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7954861, upper bound: 339.7889991
time: 1.07 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -84.7543106, 282.9392395, -86.9016266, 290.9729919, -375.7272949, 369.8408813
1: -118.9190826, 280.8794250, -121.9671478, 288.7713013, -407.6903381, 402.8465576
2: -100.8788681, 309.3803711, -103.4557571, 318.0245056, -418.9033203, 412.8361206
3: -105.7868347, 402.0895996, -108.5039368, 413.2884216, -519.0751953, 510.5935364
4: -90.3386993, 365.6924744, -92.6408310, 375.7300415, -466.0687256, 458.3333130

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8001432, upper bound: 339.7968289
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7973314, upper bound: 339.7964998
time: 1.00 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -82.0083618, 276.1442871, -79.8390961, 269.9017029, -351.9100647, 355.9833984
1: -115.1273270, 273.8354492, -111.9620819, 267.5231323, -382.6504517, 385.7975464
2: -97.5897751, 301.6179199, -94.8812561, 294.6864319, -392.2761841, 396.4991760
3: -102.4579239, 392.4111328, -99.7105865, 383.5203247, -485.9782410, 492.1217041
4: -87.4733658, 356.5697021, -85.1273880, 348.4832153, -435.9565735, 441.6970825

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7960006, upper bound: 339.7882531
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7934281, upper bound: 339.7878935
time: 2.19 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -85.0195847, 285.5716248, -86.8206635, 290.9613037, -375.9808960, 372.3922424
1: -119.4394302, 283.2567139, -121.8852234, 288.7020874, -408.1415100, 405.1419373
2: -101.2717285, 311.9241943, -103.3749008, 317.9341125, -419.2057800, 415.2990417
3: -106.2479553, 405.6071167, -108.4282227, 413.2517700, -519.4997559, 514.0353394
4: -90.7126236, 368.6193542, -92.5746994, 375.6607361, -466.3733521, 461.1940308

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7983500, upper bound: 339.7957137
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7953131, upper bound: 339.7953131
time: 1.03 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.63 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.63
Output dim: 0, lower bound: -339.7980131, upper bound: 339.7892662
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.63
Output dim: 0, lower bound: -339.7954861, upper bound: 339.7889991
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.63
Output dim: 0, lower bound: -339.8001432, upper bound: 339.7968289
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.63
Output dim: 0, lower bound: -339.7973314, upper bound: 339.7964998
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.63
Output dim: 0, lower bound: -339.7960006, upper bound: 339.7882531
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.63
Output dim: 0, lower bound: -339.7934281, upper bound: 339.7878935
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.63
Output dim: 0, lower bound: -339.7983500, upper bound: 339.7957137
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.63
Output dim: 0, lower bound: -339.7953131, upper bound: 339.7953131

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -78.1280518, 261.1025696, -77.8250046, 262.7816162, -340.9096069, 338.9275818
1: -109.5761261, 259.1481018, -109.1401138, 260.5019226, -370.0780640, 368.2881775
2: -92.9573212, 285.4800110, -92.5069199, 286.9611206, -379.9183960, 377.9868774
3: -97.5108337, 371.0456848, -97.2011337, 373.3202820, -470.8310852, 468.2468262
4: -83.3113098, 337.5531006, -82.9990082, 339.2907410, -422.6020508, 420.5521240

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7772076
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7848716, upper bound: 339.7743786
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -85.3687668, 284.2007446, -77.4837952, 261.6145630, -346.9833374, 361.6845398
1: -119.7983246, 282.1123047, -108.5861282, 259.3461304, -379.1444702, 390.6984253
2: -101.6147385, 310.8825378, -92.0355682, 285.7358398, -387.3505554, 402.9180908
3: -106.5386734, 403.6206970, -96.7148666, 371.7478333, -478.2864990, 500.3355713
4: -90.9186859, 367.5362549, -82.5848312, 337.8798828, -428.7985840, 450.1210632

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7911835, upper bound: 339.7844711
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7913170, upper bound: 339.7831869
time: 1.37 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -81.4707489, 271.4189453, -85.1939621, 285.0198059, -366.4905396, 356.6129150
1: -114.3069229, 269.4750061, -119.5703506, 282.8655396, -397.1724548, 389.0453491
2: -97.0015640, 296.8183594, -101.4350967, 311.5183411, -408.5198975, 398.2534485
3: -101.6782150, 385.5042725, -106.3696823, 404.7261658, -506.4043884, 491.8739624
4: -86.8741684, 350.7460022, -90.8368225, 368.0035095, -454.8776855, 441.5827637

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882130, upper bound: 339.7887020
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882090, upper bound: 339.7857735
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -88.6003113, 294.0981140, -84.5042648, 282.5668335, -371.1671448, 378.6023560
1: -124.3464966, 292.0219421, -118.5358124, 280.4632568, -404.8097229, 410.5577393
2: -105.5056076, 321.7790527, -100.5638580, 308.9351807, -414.4407959, 422.3428650
3: -110.5420532, 417.4205017, -105.4592056, 401.3002014, -511.8422241, 522.8796387
4: -94.3431854, 380.1632385, -90.0608444, 364.9497375, -459.2929077, 470.2240906

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7875032, upper bound: 339.7895074
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7869805, upper bound: 339.7855956
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -78.6662521, 264.5134888, -77.9937592, 263.4797974, -342.1459961, 342.5072632
1: -110.4637985, 262.3036499, -109.3839951, 261.1549377, -371.6187134, 371.6875305
2: -93.6492615, 288.8800964, -92.7037430, 287.6686401, -381.3178711, 381.5838318
3: -98.3026810, 375.6695557, -97.4134903, 374.2916565, -472.5943298, 473.0830383
4: -83.9516373, 341.4743347, -83.1762009, 340.1624146, -424.1140442, 424.6505432

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7821149, upper bound: 339.7776998
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7819492, upper bound: 339.7728449
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -85.7091370, 286.9854126, -77.6398621, 262.2712097, -347.9803162, 364.6252747
1: -120.4225616, 284.6623535, -108.8147964, 259.9593506, -380.3818970, 393.4771118
2: -102.0788269, 313.6376953, -92.2202606, 286.3943787, -388.4731750, 405.8579102
3: -107.0852890, 407.3318481, -96.9140320, 372.6676636, -479.7529297, 504.2458801
4: -91.3556900, 370.6802063, -82.7547836, 338.7039490, -430.0596313, 453.4349670

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7875769, upper bound: 339.7682612
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7875769, upper bound: 339.7859559
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -81.7210388, 274.0747681, -85.1108627, 285.0090637, -366.7301025, 359.1856079
1: -114.8246613, 271.8543091, -119.4868011, 282.7951050, -397.6197510, 391.3411255
2: -97.3763428, 299.3611450, -101.3519287, 311.4254150, -408.8016968, 400.7130737
3: -102.1364441, 389.0834656, -106.2924271, 404.6925049, -506.8289490, 495.3758850
4: -87.2344513, 353.7125244, -90.7694321, 367.9362183, -455.1706543, 444.4819336

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7851191, upper bound: 339.7886074
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7846034, upper bound: 339.7829268
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -88.8756104, 296.7409058, -84.4125519, 282.5387268, -371.4143066, 381.1534424
1: -124.8862534, 294.4281006, -118.4415054, 280.3682251, -405.2544861, 412.8695984
2: -105.8975830, 324.3549500, -100.4697647, 308.8128662, -414.7104492, 424.8247070
3: -111.0153580, 421.0601807, -105.3719025, 401.2450867, -512.2604370, 526.4319458
4: -94.7177429, 383.1982727, -89.9829559, 364.8592224, -459.5769653, 473.1812134

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7839195, upper bound: 339.7887497
time: 1.15 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7827062, upper bound: 339.7827062
time: 1.16 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.87 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7772076
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 0, lower bound: -339.7848716, upper bound: 339.7743786
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 0, lower bound: -339.7911835, upper bound: 339.7844711
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 0, lower bound: -339.7913170, upper bound: 339.7831869
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 0, lower bound: -339.7882130, upper bound: 339.7887020
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 0, lower bound: -339.7882090, upper bound: 339.7857735
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 0, lower bound: -339.7875032, upper bound: 339.7895074
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 0, lower bound: -339.7869805, upper bound: 339.7855956
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 0, lower bound: -339.7821149, upper bound: 339.7776998
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 0, lower bound: -339.7819492, upper bound: 339.7728449
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 0, lower bound: -339.7875769, upper bound: 339.7682612
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 0, lower bound: -339.7875769, upper bound: 339.7859559
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 0, lower bound: -339.7851191, upper bound: 339.7886074
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 0, lower bound: -339.7846034, upper bound: 339.7829268
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 0, lower bound: -339.7839195, upper bound: 339.7887497
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.87
Output dim: 0, lower bound: -339.7827062, upper bound: 339.7827062

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -77.1865997, 258.1778870, -69.8890686, 236.7590179, -313.9456177, 328.0669556
1: -108.3108902, 256.2066956, -98.1335449, 234.7109375, -343.0218201, 354.3402405
2: -91.8771667, 282.2496948, -83.1299591, 258.5711975, -350.4483643, 365.3795471
3: -96.3768845, 366.8483582, -87.3951035, 336.4742737, -432.8511658, 454.2434692
4: -82.3424377, 333.7400208, -74.6513138, 305.7756653, -388.1181030, 408.3912964

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7743786
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7743786
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -76.1348801, 254.7098236, -73.4923401, 248.9361267, -325.0710144, 328.2021484
1: -106.6816483, 252.7738342, -102.8450089, 246.7056427, -353.3872070, 355.6188354
2: -90.4912033, 278.4561768, -87.1579819, 271.7977905, -362.2888794, 365.6141357
3: -94.9504623, 362.0514221, -91.6403961, 353.8218384, -448.7722473, 453.6918335
4: -81.1195450, 329.3020020, -78.3563995, 321.4367371, -402.5562744, 407.6583862

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7848716, upper bound: 339.7743786
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7848716, upper bound: 339.7743786
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -84.1990356, 280.0802612, -72.7244110, 244.2526855, -328.4517212, 352.8046875
1: -118.1441803, 278.0551453, -101.1767426, 242.2502747, -360.3944702, 379.2318420
2: -100.2214279, 306.4390564, -85.8207245, 267.0136414, -367.2350769, 392.2597656
3: -105.0674896, 397.7427979, -90.2120590, 347.3356628, -452.4031372, 487.9548035
4: -89.6755829, 362.2166748, -77.2648468, 315.6214905, -405.2970276, 439.4814758

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7911835, upper bound: 339.7844711
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7911835, upper bound: 339.7844157
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -82.4874268, 274.7872009, -72.0249329, 243.2354431, -325.7228699, 346.8121338
1: -115.6719360, 273.0147400, -100.6388702, 241.5970306, -357.2689209, 373.6535645
2: -98.0667572, 300.9593506, -85.2589340, 266.3725891, -364.4393311, 386.2182922
3: -102.9308777, 390.6963196, -89.7442093, 346.4095459, -449.3404236, 480.4404907
4: -87.8271637, 355.7880554, -76.7078094, 314.9570923, -402.7842407, 432.4958496

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7866702, upper bound: 339.7689132
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7793804, upper bound: 339.7689987
time: 1.43 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -80.5739365, 268.5728760, -77.2388077, 258.7796326, -339.3535461, 345.8116760
1: -113.0775757, 266.6220093, -108.4342346, 256.8646851, -369.9422607, 375.0562134
2: -95.9521561, 293.6831970, -91.9490204, 282.8911438, -378.8432922, 385.6321716
3: -100.5766068, 381.4378357, -96.4523544, 367.5659790, -468.1425781, 477.8901978
4: -85.9323959, 347.0379944, -82.3822174, 334.1601257, -420.0925293, 429.4202271

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882090, upper bound: 339.7857735
time: 1.50 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882090, upper bound: 339.7857735
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -79.4504013, 265.0643921, -80.8371887, 271.3254700, -350.7758789, 345.9015808
1: -111.4365540, 263.1488647, -113.3423386, 269.2051086, -380.6416321, 376.4912109
2: -94.5561218, 289.8324890, -96.1288757, 296.4541626, -391.0102844, 385.9613342
3: -99.1382751, 376.5728149, -100.8625031, 385.4335327, -484.5718079, 477.4353027
4: -84.6979980, 342.5524902, -86.1443176, 350.2882385, -434.9862061, 428.6968079

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882090, upper bound: 339.7857735
time: 1.33 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882090, upper bound: 339.7857735
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -87.6993408, 291.1416321, -76.7873001, 257.0617676, -344.7610474, 367.9289246
1: -123.0966949, 289.0894775, -107.7426682, 255.2671661, -378.3638611, 396.8320618
2: -104.4429550, 318.5527954, -91.3634567, 281.1837463, -385.6267090, 409.9162292
3: -109.4270248, 413.2093811, -95.8507233, 365.1706238, -474.5976257, 509.0601196
4: -93.3922043, 376.3278809, -81.8646545, 332.0798035, -425.4720154, 458.1925354

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7544257, upper bound: 339.7791630
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7544257, upper bound: 339.7895074
time: 1.34 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -86.4008408, 287.2820740, -80.3030853, 269.4192810, -355.8200684, 367.5851440
1: -121.2772598, 285.2433167, -112.5053482, 267.3439026, -388.6211548, 397.7486572
2: -102.8878098, 314.2826233, -95.4234238, 294.4872437, -397.3750000, 409.7060242
3: -107.8213882, 407.8549805, -100.1254425, 382.7691956, -490.5905762, 507.9804077
4: -92.0119400, 371.3875122, -85.5353546, 347.9388123, -439.9507446, 456.9228210

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7535217, upper bound: 339.7765961
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7535217, upper bound: 339.7855956
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -77.7327271, 261.6315613, -70.0042496, 237.2690125, -315.0017395, 331.6357727
1: -109.2232132, 259.4067383, -98.3050842, 235.1760712, -344.3992920, 357.7117920
2: -92.5896072, 285.6957092, -83.2631989, 259.0765381, -351.6661377, 368.9589233
3: -97.1900253, 371.5373840, -87.5442429, 337.1849365, -434.3749695, 459.0816345
4: -83.0000229, 337.7205505, -74.7724304, 306.4169617, -389.4169922, 412.4929810

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7731071, upper bound: 339.7776473
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7731071, upper bound: 339.7776998
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -76.7579269, 258.4104919, -73.6251831, 249.5441895, -326.3020325, 332.0356750
1: -107.6806030, 256.2134399, -103.0392685, 247.2622070, -354.9428101, 359.2527161
2: -91.2799683, 282.1707458, -87.3171997, 272.4046021, -363.6845703, 369.4878540
3: -95.8455811, 367.0816345, -91.8106461, 354.6626587, -450.5082397, 458.8922729
4: -81.8644028, 333.5957336, -78.5043030, 322.1894531, -404.0538635, 412.1000366

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727616, upper bound: 339.7701698
time: 1.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727616, upper bound: 339.7728449
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -85.7091370, 286.9854126, -74.3630905, 250.5163879, -336.2255249, 361.3485107
1: -120.4225616, 284.6623535, -104.1736298, 248.4224548, -368.8449707, 388.8359985
2: -102.0788269, 313.6376953, -88.3227921, 273.7520142, -375.8308411, 401.9604797
3: -107.0852890, 407.3318481, -92.7924728, 356.0406189, -463.1259155, 500.1243286
4: -91.3556900, 370.6802063, -79.2654953, 323.6828613, -415.0385437, 449.9457092

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7791630, upper bound: 339.7550449
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7765961, upper bound: 339.7543485
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -85.7091370, 286.9854126, -75.6028824, 255.9156342, -341.6247559, 362.5882874
1: -120.4225616, 284.6623535, -106.0380173, 253.5903778, -374.0129395, 390.7003784
2: -102.0788269, 313.6376953, -89.8477097, 279.3692322, -381.4480286, 403.4854126
3: -107.0852890, 407.3318481, -94.4356232, 363.6600952, -470.7453613, 501.7674255
4: -91.3556900, 370.6802063, -80.6520309, 330.4671936, -421.8228760, 451.3322144

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7791630, upper bound: 339.7706597
time: 1.28 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7765961, upper bound: 339.7702602
time: 1.20 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -80.8181915, 271.2183838, -77.1780243, 258.8630981, -339.6812744, 348.3963928
1: -113.5924377, 268.9949951, -108.3860779, 256.8720398, -370.4644775, 377.3810425
2: -96.3256226, 296.2127380, -91.8909302, 282.8866882, -379.2123108, 388.1036682
3: -101.0337830, 385.0019531, -96.4051208, 367.6581421, -468.6918945, 481.4070435
4: -86.2919846, 349.9974670, -82.3360748, 334.2130737, -420.5050659, 432.3335571

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7731071, upper bound: 339.7837276
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7731071, upper bound: 339.7886074
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -79.7640991, 267.9353027, -80.7506561, 271.2968750, -351.0608215, 348.6859436
1: -112.0247650, 265.7312622, -113.2492294, 269.1157837, -381.1405334, 378.9803772
2: -94.9906693, 292.6023865, -96.0391769, 296.3461914, -391.3367920, 388.6415710
3: -99.6615829, 380.4403687, -100.7775726, 385.3719177, -485.0335083, 481.2179260
4: -85.1123581, 345.7667236, -86.0740814, 350.2055359, -435.3179016, 431.8408203

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727616, upper bound: 339.7802085
time: 1.48 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7727616, upper bound: 339.7829268
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -87.9832001, 293.8407898, -76.7282715, 257.1683960, -345.1516113, 370.5690613
1: -123.6510468, 291.5499878, -107.6998215, 255.2983704, -378.9494019, 399.2498169
2: -104.8467560, 321.1803284, -91.3099594, 281.2040100, -386.0507507, 412.4902954
3: -109.9138870, 416.9273987, -95.8086853, 365.3067017, -475.2205505, 512.7360229
4: -93.7770004, 379.4263306, -81.8233414, 332.1699524, -425.9469604, 461.2496643

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7544257, upper bound: 339.7861242
time: 1.12 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7717801, upper bound: 339.7887497
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -86.8031998, 290.3961182, -80.2202530, 269.4054871, -356.2086792, 370.6163330
1: -121.9912186, 288.1057434, -112.4183807, 267.2656860, -389.2568970, 400.5240784
2: -103.4253235, 317.3648682, -95.3375778, 294.3895569, -397.8148804, 412.7024536
3: -108.4487457, 412.1466370, -100.0464172, 382.7599792, -491.2087097, 512.1930542
4: -92.5174179, 375.0015259, -85.4683456, 347.8916016, -440.4090271, 460.4698486

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7706997, upper bound: 339.7806504
time: 1.34 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7706997, upper bound: 339.7827062
time: 1.42 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.88 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7743786
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7743786
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7848716, upper bound: 339.7743786
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7848716, upper bound: 339.7743786
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7911835, upper bound: 339.7844711
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7911835, upper bound: 339.7844157
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7866702, upper bound: 339.7689132
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7793804, upper bound: 339.7689987
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7882090, upper bound: 339.7857735
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7882090, upper bound: 339.7857735
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7882090, upper bound: 339.7857735
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7882090, upper bound: 339.7857735
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7544257, upper bound: 339.7791630
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7544257, upper bound: 339.7895074
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7535217, upper bound: 339.7765961
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7535217, upper bound: 339.7855956
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7731071, upper bound: 339.7776473
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7731071, upper bound: 339.7776998
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7727616, upper bound: 339.7701698
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7727616, upper bound: 339.7728449
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7791630, upper bound: 339.7550449
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7765961, upper bound: 339.7543485
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7791630, upper bound: 339.7706597
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7765961, upper bound: 339.7702602
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7731071, upper bound: 339.7837276
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7731071, upper bound: 339.7886074
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7727616, upper bound: 339.7802085
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7727616, upper bound: 339.7829268
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7544257, upper bound: 339.7861242
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7717801, upper bound: 339.7887497
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7706997, upper bound: 339.7806504
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 0, lower bound: -339.7706997, upper bound: 339.7827062

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -69.9816055, 234.2883453, -69.8890686, 236.7590179, -306.7406006, 304.1774292
1: -98.1983566, 232.5957642, -98.1335449, 234.7109375, -332.9093018, 330.7293091
2: -83.2672043, 256.2710876, -83.1299591, 258.5711975, -341.8384094, 339.4008789
3: -87.3797989, 333.1095276, -87.3951035, 336.4742737, -423.8540649, 420.5046387
4: -74.6845932, 303.0175171, -74.6513138, 305.7756653, -380.4602661, 377.6687622

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7772076
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7772076
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -73.7124176, 246.9560547, -69.8890686, 236.7590179, -310.4714355, 316.8451233
1: -103.1775589, 245.0378113, -98.1335449, 234.7109375, -337.8884888, 343.1713562
2: -87.5162354, 269.9715271, -83.1299591, 258.5711975, -346.0874023, 353.1013489
3: -91.8525848, 351.1326294, -87.3951035, 336.4742737, -428.3268433, 438.5277405
4: -78.5416183, 319.3222961, -74.6513138, 305.7756653, -384.3172913, 393.9735718

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7772076
time: 1.45 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7772076
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -69.9816055, 234.2883453, -73.4923401, 248.9361267, -318.9177246, 307.7807007
1: -98.1983566, 232.5957642, -102.8450089, 246.7056427, -344.9039917, 335.4407654
2: -83.2672043, 256.2710876, -87.1579819, 271.7977905, -355.0650024, 343.4289551
3: -87.3797989, 333.1095276, -91.6403961, 353.8218384, -441.2016296, 424.7499390
4: -74.6845932, 303.0175171, -78.3563995, 321.4367371, -396.1213074, 381.3739014

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7743786
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7743786
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -73.7124176, 246.9560547, -73.4923401, 248.9361267, -322.6485596, 320.4483948
1: -103.1775589, 245.0378113, -102.8450089, 246.7056427, -349.8831787, 347.8828125
2: -87.5162354, 269.9715271, -87.1579819, 271.7977905, -359.3139954, 357.1294556
3: -91.8525848, 351.1326294, -91.6403961, 353.8218384, -445.6744385, 442.7730103
4: -78.5416183, 319.3222961, -78.3563995, 321.4367371, -399.9783630, 397.6786804

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7743786
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7743786
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -74.2157211, 248.3571777, -71.4829254, 240.0093994, -314.2251282, 319.8400574
1: -103.9944611, 246.4121857, -99.4585190, 238.0619659, -342.0564270, 345.8706970
2: -88.2391052, 271.6640320, -84.3810730, 262.4057007, -350.6447144, 356.0451050
3: -92.5167084, 353.0584412, -88.6775055, 341.2903442, -433.8070374, 441.7359619
4: -79.0438080, 321.4393005, -75.9673767, 310.1308594, -389.1746216, 397.4066772

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7911300, upper bound: 339.7844711
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7911300, upper bound: 339.7844711
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -82.2513962, 273.5397339, -71.9302444, 241.6131439, -323.8645325, 345.4699097
1: -115.3736725, 271.5705261, -100.0311279, 239.6260071, -354.9996948, 371.6016541
2: -97.8724060, 299.3164673, -84.8524017, 264.1384888, -362.0108948, 384.1688843
3: -102.6134415, 388.3642883, -89.2000351, 343.5677490, -446.1811829, 477.5643311
4: -87.5559692, 353.7118225, -76.4118195, 312.1934509, -399.7494202, 430.1236572

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7911300, upper bound: 339.7844157
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7911300, upper bound: 339.7844157
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -75.2960434, 251.3963776, -71.0183105, 240.1864014, -315.4824524, 322.4146729
1: -105.5979843, 249.7952881, -99.3509903, 238.5546722, -344.1526489, 349.1462708
2: -89.4948349, 275.3491516, -84.1492462, 262.9883728, -352.4832153, 359.4984131
3: -93.9700623, 357.5126343, -88.5868378, 342.0713196, -436.0413818, 446.0994873
4: -80.1984634, 325.4843750, -75.6545410, 310.9882202, -391.1866760, 401.1389160

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7812315, upper bound: 339.7533446
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7812315, upper bound: 339.7689132
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -77.9220047, 260.1729431, -70.0213470, 236.8657684, -314.7877808, 330.1942749
1: -109.0790939, 258.5470276, -97.7248077, 235.2572327, -344.3363342, 356.2718506
2: -92.4558640, 284.9891663, -82.7847443, 259.4199829, -351.8757935, 367.7738953
3: -97.0910416, 370.2235107, -87.1673508, 337.4555054, -434.5465393, 457.3908386
4: -82.8965225, 337.0310059, -74.5745316, 306.7628174, -389.6593323, 411.6055298

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7394987, upper bound: 339.7541183
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7391007, upper bound: 339.7421127
time: 1.39 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -73.3717346, 244.7259064, -77.2388077, 258.7796326, -332.1513672, 321.9647217
1: -102.9647675, 243.0084534, -108.4342346, 256.8646851, -359.8294678, 351.4426575
2: -87.3470001, 267.6911621, -91.9490204, 282.8911438, -370.2381287, 359.6401672
3: -91.5791779, 347.6546021, -96.4523544, 367.5659790, -459.1451416, 444.1069641
4: -78.2764740, 316.3160095, -82.3822174, 334.1601257, -412.4365845, 398.6982422

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882130, upper bound: 339.7887020
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882130, upper bound: 339.7887020
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -77.0213852, 257.2641296, -77.2388077, 258.7796326, -335.8009949, 334.5029297
1: -107.9215546, 255.3678589, -108.4342346, 256.8646851, -364.7862549, 363.8020325
2: -91.5642624, 281.2699585, -91.9490204, 282.8911438, -374.4553223, 373.2189331
3: -96.0283508, 365.5830383, -96.4523544, 367.5659790, -463.5942993, 462.0354004
4: -82.0699615, 332.4961548, -82.3822174, 334.1601257, -416.2301025, 414.8783569

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882130, upper bound: 339.7887020
time: 1.34 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882130, upper bound: 339.7887020
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -73.3717346, 244.7259064, -80.8371887, 271.3254700, -344.6972046, 325.5630798
1: -102.9647675, 243.0084534, -113.3423386, 269.2051086, -372.1698608, 356.3507385
2: -87.3470001, 267.6911621, -96.1288757, 296.4541626, -383.8011475, 363.8200378
3: -91.5791779, 347.6546021, -100.8625031, 385.4335327, -477.0126953, 448.5170898
4: -78.2764740, 316.3160095, -86.1443176, 350.2882385, -428.5646973, 402.4603271

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882090, upper bound: 339.7857735
time: 1.44 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882090, upper bound: 339.7857735
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -77.0213852, 257.2641296, -80.8371887, 271.3254700, -348.3468628, 338.1013184
1: -107.9215546, 255.3678589, -113.3423386, 269.2051086, -377.1266479, 368.7101440
2: -91.5642624, 281.2699585, -96.1288757, 296.4541626, -388.0183716, 377.3988342
3: -96.0283508, 365.5830383, -100.8625031, 385.4335327, -481.4617920, 466.4455566
4: -82.0699615, 332.4961548, -86.1443176, 350.2882385, -432.3581543, 418.6404724

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882090, upper bound: 339.7857735
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882090, upper bound: 339.7857735
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -80.1561432, 268.8508301, -76.7873001, 257.0617676, -337.2178955, 345.6381226
1: -112.5410690, 266.6344604, -107.7426682, 255.2671661, -367.8082275, 374.3771057
2: -95.4238205, 293.8865967, -91.3634567, 281.1837463, -376.6075745, 385.2500305
3: -100.1338882, 381.6705322, -95.8507233, 365.1706238, -465.3045044, 477.5212402
4: -85.4590530, 347.3961487, -81.8646545, 332.0798035, -417.5388489, 429.2608032

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7540831, upper bound: 339.7791630
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7540831, upper bound: 339.7791630
time: 1.27 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -87.1872482, 289.2525635, -76.7873001, 257.0617676, -344.2489624, 366.0398560
1: -122.3676224, 287.2465210, -107.7426682, 255.2671661, -377.6347961, 394.9891663
2: -103.8317184, 316.5340881, -91.3634567, 281.1837463, -385.0154724, 407.8975220
3: -108.7779465, 410.5479431, -95.8507233, 365.1706238, -473.9485779, 506.3986816
4: -92.8461838, 373.9342651, -81.8646545, 332.0798035, -424.9259949, 455.7988892

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7540831, upper bound: 339.7895074
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7540831, upper bound: 339.7895074
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -78.8873520, 265.0354004, -80.3030853, 269.4192810, -348.3065796, 345.3385010
1: -110.7369003, 262.8562317, -112.5053482, 267.3439026, -378.0808105, 375.3615723
2: -93.8825226, 289.6978760, -95.4234238, 294.4872437, -388.3697510, 385.1212769
3: -98.5458603, 376.3816528, -100.1254425, 382.7691956, -481.3150635, 476.5070801
4: -84.0981445, 342.5168762, -85.5353546, 347.9388123, -432.0369568, 428.0521851

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7432481, upper bound: 339.7658741
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7535217, upper bound: 339.7765961
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7535217, upper bound: 339.7765961
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -85.8347855, 285.2328796, -80.3030853, 269.4192810, -355.2540588, 365.5359497
1: -120.4808884, 283.2369690, -112.5053482, 267.3439026, -387.8247986, 395.7423096
2: -102.2194443, 312.0818176, -95.4234238, 294.4872437, -396.7066650, 407.5051880
3: -107.1108170, 404.9671326, -100.1254425, 382.7691956, -489.8800049, 505.0925903
4: -91.4138718, 368.7904358, -85.5353546, 347.9388123, -439.3526917, 454.3257141

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7432481, upper bound: 339.7583757
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7535217, upper bound: 339.7855956
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7535217, upper bound: 339.7855956
time: 1.20 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 7.26 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7772076
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7772076
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7772076
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7772076
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7743786
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7743786
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7743786
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7847347, upper bound: 339.7743786
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7911300, upper bound: 339.7844711
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7911300, upper bound: 339.7844711
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7911300, upper bound: 339.7844157
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7911300, upper bound: 339.7844157
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7812315, upper bound: 339.7533446
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7812315, upper bound: 339.7689132
IS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7394987, upper bound: 339.7541183
IS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7391007, upper bound: 339.7421127
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7882130, upper bound: 339.7887020
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7882130, upper bound: 339.7887020
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7882130, upper bound: 339.7887020
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7882130, upper bound: 339.7887020
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7882090, upper bound: 339.7857735
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7882090, upper bound: 339.7857735
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7882090, upper bound: 339.7857735
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7882090, upper bound: 339.7857735
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7540831, upper bound: 339.7791630
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7540831, upper bound: 339.7791630
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7540831, upper bound: 339.7895074
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7540831, upper bound: 339.7895074
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7535217, upper bound: 339.7765961
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7535217, upper bound: 339.7765961
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7535217, upper bound: 339.7855956
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.26
Output dim: 0, lower bound: -339.7535217, upper bound: 339.7855956
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 0, lower bound: -339.7731071, upper bound: 339.7776473
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 0, lower bound: -339.7731071, upper bound: 339.7776998
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 0, lower bound: -339.7727616, upper bound: 339.7701698
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 0, lower bound: -339.7727616, upper bound: 339.7728449
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 0, lower bound: -339.7791630, upper bound: 339.7550449
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 0, lower bound: -339.7765961, upper bound: 339.7543485
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 0, lower bound: -339.7791630, upper bound: 339.7706597
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 0, lower bound: -339.7765961, upper bound: 339.7702602
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 0, lower bound: -339.7731071, upper bound: 339.7837276
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 0, lower bound: -339.7731071, upper bound: 339.7886074
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 0, lower bound: -339.7727616, upper bound: 339.7802085
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 0, lower bound: -339.7727616, upper bound: 339.7829268
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 0, lower bound: -339.7544257, upper bound: 339.7861242
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 0, lower bound: -339.7717801, upper bound: 339.7887497
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 0, lower bound: -339.7706997, upper bound: 339.7806504
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.26
Output dim: 0, lower bound: -339.7706997, upper bound: 339.7827062
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=385.80084228515625
rel_dist={0: [-339.8051238459851, 339.80512384598524]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1103.63 seconds
