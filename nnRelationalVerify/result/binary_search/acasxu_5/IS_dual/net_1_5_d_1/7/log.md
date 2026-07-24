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
execution time: IAR + LP analysis = 2.36 + 2.38 = 4.74 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -339.8056876, upper bound: 339.8056876


# Binary Search by BASE starts (time budget: 1195.26 seconds, max iter: 100)

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
Binary search time: 90.73 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1104.53 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8056352, upper bound: 339.8042672
time: 1.19 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
time: 0.97 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.36 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 2.36
Output dim: 0, lower bound: -339.8056352, upper bound: 339.8042672
IS_B2, status: Status.UNKNOWN, split count: 1, time: 2.36
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -88.6961975, 297.1046753, -85.0949707, 284.1704712, -372.8666382, 382.1996460
1: -124.4471970, 294.8176575, -119.3920059, 282.0884705, -406.5356750, 414.2096252
2: -105.5478058, 324.6724243, -101.2751236, 310.7092590, -416.2570496, 425.9475098
3: -110.7164154, 421.9519958, -106.2089005, 403.8296509, -514.5460205, 528.1608887
4: -94.5076294, 383.5692749, -90.6926956, 367.2429504, -461.7505798, 474.2619629

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7953359, upper bound: 339.7687211
time: 0.96 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7672412, upper bound: 339.7653524
time: 1.08 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -88.6961975, 297.1046753, -85.5067825, 287.2420959, -375.9382324, 382.6114502
1: -124.4471970, 294.8176575, -120.1114273, 284.9082947, -409.3554993, 414.9290466
2: -105.5478058, 324.6724243, -101.8361511, 313.7402954, -419.2880859, 426.5085449
3: -110.7164154, 421.9519958, -106.8467102, 407.9730225, -518.6894531, 528.7987061
4: -94.5076294, 383.5692749, -91.2148666, 370.7578735, -465.2655029, 474.7841492

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7708367, upper bound: 339.7958408
time: 1.01 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7674305, upper bound: 339.7674305
time: 0.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.35 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 4.35
Output dim: 0, lower bound: -339.7953359, upper bound: 339.7687211
IS_B1_A2, status: Status.VERIFIED, split count: 2, time: 4.35
Output dim: 0, lower bound: -339.7672412, upper bound: 339.7653524
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 4.35
Output dim: 0, lower bound: -339.7708367, upper bound: 339.7958408
IS_B2_B2, status: Status.VERIFIED, split count: 2, time: 4.35
Output dim: 0, lower bound: -339.7674305, upper bound: 339.7674305

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -86.7272873, 290.3902588, -85.0949707, 284.1704712, -370.8977661, 375.4852295
1: -121.6965637, 288.1918945, -119.3920059, 282.0884705, -403.7850037, 407.5838928
2: -103.2365189, 317.3988342, -101.2751236, 310.7092590, -413.9457703, 418.6739197
3: -108.2730484, 412.3710938, -106.2089005, 403.8296509, -512.1027222, 518.5800171
4: -92.4498215, 374.8715820, -90.6926956, 367.2429504, -459.6927795, 465.5642700

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7951081, upper bound: 339.7681463
time: 0.77 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7937694, upper bound: 339.7679594
time: 0.96 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -88.6961975, 297.1046753, -83.4957504, 280.3916321, -369.0877991, 380.6003723
1: -124.4471970, 294.8176575, -117.3040619, 278.1459656, -402.5931702, 412.1217041
2: -105.5478058, 324.6724243, -99.4742203, 306.3189087, -411.8666992, 424.1466370
3: -110.7164154, 421.9519958, -104.3519974, 398.1978760, -508.9143066, 526.3040161
4: -94.5076294, 383.5692749, -89.1125412, 361.8850098, -456.3926392, 472.6817627

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_B1

### Relational analysis result of IS_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7702512, upper bound: 339.7953963
time: 1.01 seconds

## Relational analysis of IS_B2_B1_B2

### Relational analysis result of IS_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7700375, upper bound: 339.7939588
time: 1.22 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.21 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 5.21
Output dim: 0, lower bound: -339.7951081, upper bound: 339.7681463
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 5.21
Output dim: 0, lower bound: -339.7937694, upper bound: 339.7679594
IS_B2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 5.21
Output dim: 0, lower bound: -339.7702512, upper bound: 339.7953963
IS_B2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 5.21
Output dim: 0, lower bound: -339.7700375, upper bound: 339.7939588

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -83.7281189, 279.8192444, -85.0949707, 284.1704712, -367.8985901, 364.9141541
1: -117.4778519, 277.7323303, -119.3920059, 282.0884705, -399.5663147, 397.1242981
2: -99.6791382, 305.8737793, -101.2751236, 310.7092590, -410.3883972, 407.1488647
3: -104.5141068, 397.1990662, -106.2089005, 403.8296509, -508.3437500, 503.4079590
4: -89.2727966, 361.1928406, -90.6926956, 367.2429504, -456.5157471, 451.8855286

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A1_A1

### Relational analysis result of IS_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921112, upper bound: 339.7648440
time: 0.80 seconds

## Relational analysis of IS_B1_A1_A1_A2

### Relational analysis result of IS_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882861, upper bound: 339.7641483
time: 0.83 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -91.0158844, 302.9884033, -85.0949707, 284.1704712, -375.1863403, 388.0833740
1: -127.7341919, 300.7621155, -119.3920059, 282.0884705, -409.8226624, 420.1540833
2: -108.3706284, 331.3544312, -101.2751236, 310.7092590, -419.0798950, 432.6295166
3: -113.5672760, 429.8279419, -106.2089005, 403.8296509, -517.3968506, 536.0368652
4: -96.8971481, 391.2447815, -90.6926956, 367.2429504, -464.1401062, 481.9374695

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_A2_A1

### Relational analysis result of IS_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7937694, upper bound: 339.7679594
time: 1.01 seconds

## Relational analysis of IS_B1_A1_A2_A2

### Relational analysis result of IS_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7937694, upper bound: 339.7679594
time: 0.99 seconds

## BFS IS instance: IS_B2_B1_B1

### Backsubstitution after applying IS history:
0: -88.6961975, 297.1046753, -80.4964371, 269.8757629, -358.5719299, 377.6010742
1: -124.4471970, 294.8176575, -113.0993881, 267.7301941, -392.1773987, 407.9170532
2: -105.5478058, 324.6724243, -95.9254608, 294.8403931, -400.3881836, 420.5978088
3: -110.7164154, 421.9519958, -100.6049500, 383.0876770, -493.8040771, 522.5569458
4: -94.5076294, 383.5692749, -85.9423904, 348.2578430, -442.7654724, 469.5116272

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B2_B1_B1_B1

### Relational analysis result of IS_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7667032, upper bound: 339.7922327
time: 0.89 seconds

## Relational analysis of IS_B2_B1_B1_B2

### Relational analysis result of IS_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7659765, upper bound: 339.7884008
time: 1.28 seconds

## BFS IS instance: IS_B2_B1_B2

### Backsubstitution after applying IS history:
0: -88.6961975, 297.1046753, -87.6614304, 292.6433716, -381.3395386, 384.7661133
1: -124.4471970, 294.8176575, -123.1961212, 290.3698120, -414.8170166, 418.0137939
2: -105.5478058, 324.6724243, -104.4715958, 319.8901672, -425.4379883, 429.1439819
3: -110.7164154, 421.9519958, -109.5072021, 415.1761169, -525.8924561, 531.4592285
4: -94.5076294, 383.5692749, -93.4404907, 377.8494568, -472.3570862, 477.0097351

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B2_B1_B2_B1

### Relational analysis result of IS_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7663145, upper bound: 339.7901190
time: 1.05 seconds

## Relational analysis of IS_B2_B1_B2_B2

### Relational analysis result of IS_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7661982, upper bound: 339.7888604
time: 0.93 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.69 seconds
IS_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -339.7921112, upper bound: 339.7648440
IS_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -339.7882861, upper bound: 339.7641483
IS_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -339.7937694, upper bound: 339.7679594
IS_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -339.7937694, upper bound: 339.7679594
IS_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -339.7667032, upper bound: 339.7922327
IS_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -339.7659765, upper bound: 339.7884008
IS_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -339.7663145, upper bound: 339.7901190
IS_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.69
Output dim: 0, lower bound: -339.7661982, upper bound: 339.7888604

## BFS IS instance: IS_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -77.6765289, 258.9679260, -85.0230331, 283.9196167, -361.5960693, 343.9909363
1: -108.4467163, 257.0819702, -119.2899094, 281.8424988, -390.2892151, 376.3718872
2: -92.0782471, 283.1192932, -101.1890335, 310.4364929, -402.5147095, 384.3083191
3: -96.5577774, 367.8579407, -106.1187134, 403.4731140, -500.0308838, 473.9766541
4: -82.5927887, 334.3193054, -90.6161499, 366.9203796, -449.5131531, 424.9354553

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_A1_A1_A1

### Relational analysis result of IS_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921112, upper bound: 339.7648440
time: 0.96 seconds

## Relational analysis of IS_B1_A1_A1_A1_A2

### Relational analysis result of IS_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921112, upper bound: 339.7648440
time: 1.43 seconds

## BFS IS instance: IS_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -77.8753281, 260.1290588, -85.0949707, 284.1704712, -362.0458069, 345.2239990
1: -109.0339279, 258.6267700, -119.3920059, 282.0884705, -391.1224060, 378.0187073
2: -92.4706116, 285.0324402, -101.2751236, 310.7092590, -403.1798706, 386.3074951
3: -97.1078796, 369.9987793, -106.2089005, 403.8296509, -500.9375000, 476.2076721
4: -82.9104004, 336.5710449, -90.6926956, 367.2429504, -450.1533508, 427.2637329

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_A1_A2_A1

### Relational analysis result of IS_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882861, upper bound: 339.7641483
time: 1.23 seconds

## Relational analysis of IS_B1_A1_A1_A2_A2

### Relational analysis result of IS_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882861, upper bound: 339.7641483
time: 0.94 seconds

## BFS IS instance: IS_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -87.0811234, 288.9848938, -85.0949707, 284.1704712, -371.2515869, 374.0798645
1: -122.1800766, 287.0001221, -119.3920059, 282.0884705, -404.2684937, 406.3920898
2: -103.6976929, 316.2632751, -101.2751236, 310.7092590, -414.4069214, 417.5383301
3: -108.6225128, 410.0626221, -106.2089005, 403.8296509, -512.4521484, 516.2715454
4: -92.7320328, 373.4294434, -90.6926956, 367.2429504, -459.9749756, 464.1221313

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A2_A1_A1

### Relational analysis result of IS_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7899950, upper bound: 339.7644863
time: 1.06 seconds

## Relational analysis of IS_B1_A1_A2_A1_A2

### Relational analysis result of IS_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7887457, upper bound: 339.7643700
time: 0.88 seconds

## BFS IS instance: IS_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -87.6614304, 292.6433716, -85.0949707, 284.1704712, -371.8319092, 377.7383118
1: -123.1961212, 290.3698120, -119.3920059, 282.0884705, -405.2846069, 409.7617493
2: -104.4715958, 319.8901672, -101.2751236, 310.7092590, -415.1808472, 421.1652527
3: -109.5072021, 415.1761169, -106.2089005, 403.8296509, -513.3368530, 521.3850098
4: -93.4404907, 377.8494568, -90.6926956, 367.2429504, -460.6834106, 468.5421448

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A2_A2_A1

### Relational analysis result of IS_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7899950, upper bound: 339.7644863
time: 0.97 seconds

## Relational analysis of IS_B1_A1_A2_A2_A2

### Relational analysis result of IS_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7887457, upper bound: 339.7643700
time: 0.76 seconds

## BFS IS instance: IS_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -88.6217499, 296.8466492, -74.5175018, 249.2518616, -337.8735962, 371.3641052
1: -124.3416443, 294.5643616, -104.1531906, 247.3020935, -371.6437378, 398.7175598
2: -105.4587326, 324.3935242, -88.3960266, 272.3433533, -377.8020935, 412.7895508
3: -110.6230316, 421.5859680, -92.7213516, 354.0706177, -464.6936340, 514.3072510
4: -94.4283524, 383.2373352, -79.3333664, 321.6880798, -416.1163940, 462.5707092

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_B1_B1_B1_A1

### Relational analysis result of IS_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7648440, upper bound: 339.7921112
time: 1.00 seconds

## Relational analysis of IS_B2_B1_B1_B1_A2

### Relational analysis result of IS_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7648440, upper bound: 339.7922327
time: 1.27 seconds

## BFS IS instance: IS_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -88.6961975, 297.1046753, -74.6870956, 250.2789154, -338.9750977, 371.7917786
1: -124.4471970, 294.8176575, -104.7031937, 248.7222290, -373.1694031, 399.5208435
2: -105.5478058, 324.6724243, -88.7651215, 274.1075439, -379.6553040, 413.4375610
3: -110.7164154, 421.9519958, -93.2395935, 356.0269470, -466.7433472, 515.1915894
4: -94.5076294, 383.5692749, -79.6211777, 323.7671204, -418.2746887, 463.1904602

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_B1_B1_B2_A1

### Relational analysis result of IS_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7641483, upper bound: 339.7882861
time: 0.94 seconds

## Relational analysis of IS_B2_B1_B1_B2_A2

### Relational analysis result of IS_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7641483, upper bound: 339.7884008
time: 1.10 seconds

## BFS IS instance: IS_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -88.6217499, 296.8466492, -82.2580032, 273.6946106, -362.3163452, 379.1046448
1: -124.3416443, 294.5643616, -115.2041168, 271.6928101, -396.0344543, 409.7684937
2: -105.4587326, 324.3935242, -97.7027740, 299.2369080, -404.6956482, 422.0962830
3: -110.6230316, 421.5859680, -102.4676895, 388.6105042, -499.2335205, 524.0535889
4: -94.4283524, 383.2373352, -87.4080505, 353.4977722, -447.9261169, 470.6453552

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_B1_B2_B1_A1

### Relational analysis result of IS_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7644863, upper bound: 339.7900043
time: 1.28 seconds

## Relational analysis of IS_B2_B1_B2_B1_A2

### Relational analysis result of IS_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7644863, upper bound: 339.7901190
time: 1.07 seconds

## BFS IS instance: IS_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -88.6961975, 297.1046753, -81.8358765, 273.4550781, -362.1512756, 378.9404907
1: -124.4471970, 294.8176575, -114.8749313, 271.7871704, -396.2343750, 409.6925659
2: -105.5478058, 324.6724243, -97.3516312, 299.6034546, -405.1512451, 422.0240479
3: -110.7164154, 421.9519958, -102.2249146, 388.7427979, -499.4592285, 524.1768188
4: -94.5076294, 383.5692749, -87.1974792, 353.8367920, -448.3444214, 470.7667236

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_B1_B2_B2_A1

### Relational analysis result of IS_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7643700, upper bound: 339.7887457
time: 0.80 seconds

## Relational analysis of IS_B2_B1_B2_B2_A2

### Relational analysis result of IS_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7643700, upper bound: 339.7888604
time: 1.13 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.84 seconds
IS_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 0, lower bound: -339.7921112, upper bound: 339.7648440
IS_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 0, lower bound: -339.7921112, upper bound: 339.7648440
IS_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 0, lower bound: -339.7882861, upper bound: 339.7641483
IS_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 0, lower bound: -339.7882861, upper bound: 339.7641483
IS_B1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 0, lower bound: -339.7899950, upper bound: 339.7644863
IS_B1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 0, lower bound: -339.7887457, upper bound: 339.7643700
IS_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 0, lower bound: -339.7899950, upper bound: 339.7644863
IS_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 0, lower bound: -339.7887457, upper bound: 339.7643700
IS_B2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 0, lower bound: -339.7648440, upper bound: 339.7921112
IS_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 0, lower bound: -339.7648440, upper bound: 339.7922327
IS_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 0, lower bound: -339.7641483, upper bound: 339.7882861
IS_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 0, lower bound: -339.7641483, upper bound: 339.7884008
IS_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 0, lower bound: -339.7644863, upper bound: 339.7900043
IS_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 0, lower bound: -339.7644863, upper bound: 339.7901190
IS_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 0, lower bound: -339.7643700, upper bound: 339.7887457
IS_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 0, lower bound: -339.7643700, upper bound: 339.7888604

## BFS IS instance: IS_B1_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -74.0052567, 245.7193756, -85.0230331, 283.9196167, -357.9248657, 330.7423401
1: -103.2421646, 244.0806885, -119.2899094, 281.8424988, -385.0846558, 363.3705444
2: -87.7117767, 268.8742981, -101.1890335, 310.4364929, -398.1482544, 370.0633240
3: -91.9311142, 349.1697388, -106.1187134, 403.4731140, -495.4042053, 455.2884521
4: -78.6868591, 317.4967651, -90.6161499, 366.9203796, -445.6072388, 408.1129150

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_A1_A1_A1_A1

### Relational analysis result of IS_B1_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7914241, upper bound: 339.7597075
time: 1.32 seconds

## Relational analysis of IS_B1_A1_A1_A1_A1_A2

### Relational analysis result of IS_B1_A1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7789483, upper bound: 339.7567831
time: 0.80 seconds

## BFS IS instance: IS_B1_A1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -74.5175018, 249.2518616, -85.0230331, 283.9196167, -358.4371033, 334.2749023
1: -104.1531906, 247.3020935, -119.2899094, 281.8424988, -385.9956970, 366.5920105
2: -88.3960266, 272.3433533, -101.1890335, 310.4364929, -398.8325195, 373.5323792
3: -92.7213516, 354.0706177, -106.1187134, 403.4731140, -496.1944580, 460.1893311
4: -79.3333664, 321.6880798, -90.6161499, 366.9203796, -446.2537231, 412.3042297

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_A1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7887473, upper bound: 339.7560124
time: 0.98 seconds

## Relational analysis of IS_B1_A1_A1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7869310, upper bound: 339.7519940
time: 1.08 seconds

## BFS IS instance: IS_B1_A1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -73.8942566, 245.9733887, -85.0949707, 284.1704712, -358.0647278, 331.0683594
1: -103.4220047, 244.7231598, -119.3920059, 282.0884705, -385.5104675, 364.1151428
2: -87.7495346, 269.7869873, -101.2751236, 310.7092590, -398.4588013, 371.0620728
3: -92.1146774, 350.1335144, -106.2089005, 403.8296509, -495.9443359, 456.3424072
4: -78.7043076, 318.6690674, -90.6926956, 367.2429504, -445.9472656, 409.3617554

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_A1_A2_A1_B1

### Relational analysis result of IS_B1_A1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7835398, upper bound: 339.7551124
time: 0.93 seconds

## Relational analysis of IS_B1_A1_A1_A2_A1_B2

### Relational analysis result of IS_B1_A1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7832480, upper bound: 339.7513118
time: 0.83 seconds

## BFS IS instance: IS_B1_A1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -74.6870956, 250.2789154, -85.0949707, 284.1704712, -358.8575745, 335.3739014
1: -104.7031937, 248.7222290, -119.3920059, 282.0884705, -386.7916565, 368.1141357
2: -88.7651215, 274.1075439, -101.2751236, 310.7092590, -399.4743652, 375.3825684
3: -93.2395935, 356.0269470, -106.2089005, 403.8296509, -497.0692444, 462.2358398
4: -79.6211777, 323.7671204, -90.6926956, 367.2429504, -446.8641357, 414.4597473

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_A1_A2_A2_B1

### Relational analysis result of IS_B1_A1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7835398, upper bound: 339.7551124
time: 1.28 seconds

## Relational analysis of IS_B1_A1_A1_A2_A2_B2

### Relational analysis result of IS_B1_A1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7832480, upper bound: 339.7513118
time: 0.99 seconds

## BFS IS instance: IS_B1_A1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -82.5946960, 273.2151489, -85.0230331, 283.9196167, -366.5143127, 358.2381897
1: -115.5130005, 271.4657898, -119.2899094, 281.8424988, -397.3554993, 390.7557068
2: -98.0321426, 299.0640259, -101.1890335, 310.4364929, -408.4686279, 400.2530212
3: -102.7604294, 387.9916077, -106.1187134, 403.4731140, -506.2335510, 494.1103210
4: -87.6726685, 353.1832581, -90.6161499, 366.9203796, -454.5930481, 443.7994080

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_A2_A1_A1_A1

### Relational analysis result of IS_B1_A1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7892369, upper bound: 339.7598044
time: 1.10 seconds

## Relational analysis of IS_B1_A1_A2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_A1_A2_A1_A1_B1

### Relational analysis result of IS_B1_A1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7899950, upper bound: 339.7649837
time: 0.79 seconds

## Relational analysis of IS_B1_A1_A2_A1_A1_B2

### Relational analysis result of IS_B1_A1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7899950, upper bound: 339.7649837
time: 0.88 seconds

## BFS IS instance: IS_B1_A1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -81.1868820, 269.6222839, -85.0949707, 284.1704712, -365.3573303, 354.7172546
1: -113.7671432, 268.2580566, -119.3920059, 282.0884705, -395.8556213, 387.6500244
2: -96.4886932, 295.8059998, -101.2751236, 310.7092590, -407.1979370, 397.0810852
3: -101.2618713, 383.4673462, -106.2089005, 403.8296509, -505.0915222, 489.6762390
4: -86.4228897, 349.2984009, -90.6926956, 367.2429504, -453.6658325, 439.9910889

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_A2_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7796715, upper bound: 339.7547384
time: 1.10 seconds

## Relational analysis of IS_B1_A1_A2_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7802282, upper bound: 339.7599802
time: 0.94 seconds

## BFS IS instance: IS_B1_A1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -82.2580032, 273.6946106, -85.0230331, 283.9196167, -366.1776123, 358.7176514
1: -115.2041168, 271.6928101, -119.2899094, 281.8424988, -397.0466309, 390.9827271
2: -97.7027740, 299.2369080, -101.1890335, 310.4364929, -408.1392822, 400.4259338
3: -102.4676895, 388.6105042, -106.1187134, 403.4731140, -505.9407959, 494.7292175
4: -87.4080505, 353.4977722, -90.6161499, 366.9203796, -454.3283691, 444.1139221

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_A2_A2_A1_A1

### Relational analysis result of IS_B1_A1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7771203, upper bound: 339.7494001
time: 0.95 seconds

## Relational analysis of IS_B1_A1_A2_A2_A1_A2

### Relational analysis result of IS_B1_A1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7849361, upper bound: 339.7516390
time: 1.07 seconds

## BFS IS instance: IS_B1_A1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -81.8358765, 273.4550781, -85.0949707, 284.1704712, -366.0062866, 358.5500488
1: -114.8749313, 271.7871704, -119.3920059, 282.0884705, -396.9634094, 391.1791687
2: -97.3516312, 299.6034546, -101.2751236, 310.7092590, -408.0608826, 400.8785706
3: -102.2249146, 388.7427979, -106.2089005, 403.8296509, -506.0545654, 494.9516907
4: -87.1974792, 353.8367920, -90.6926956, 367.2429504, -454.4404297, 444.5294800

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_A2_A2_A2_B1

### Relational analysis result of IS_B1_A1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7784861, upper bound: 339.7522136
time: 1.07 seconds

## Relational analysis of IS_B1_A1_A2_A2_A2_B2

### Relational analysis result of IS_B1_A1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7790428, upper bound: 339.7574554
time: 1.15 seconds

## BFS IS instance: IS_B2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -85.0230331, 283.9196167, -74.5175018, 249.2518616, -334.2749023, 358.4371033
1: -119.2899094, 281.8424988, -104.1531906, 247.3020935, -366.5920105, 385.9956970
2: -101.1890335, 310.4364929, -88.3960266, 272.3433533, -373.5323792, 398.8325195
3: -106.1187134, 403.4731140, -92.7213516, 354.0706177, -460.1893311, 496.1944580
4: -90.6161499, 366.9203796, -79.3333664, 321.6880798, -412.3042297, 446.2537537

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B1_B1_A1_A1

### Relational analysis result of IS_B2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7560124, upper bound: 339.7887473
time: 1.00 seconds

## Relational analysis of IS_B2_B1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B1_B1_A1_B1

### Relational analysis result of IS_B2_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7637479, upper bound: 339.7898049
time: 0.97 seconds

## Relational analysis of IS_B2_B1_B1_B1_A1_B2

### Relational analysis result of IS_B2_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7637484, upper bound: 339.7897849
time: 1.25 seconds

## BFS IS instance: IS_B2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -85.4314194, 286.9815979, -74.5175018, 249.2518616, -334.6832886, 361.4990845
1: -120.0044708, 284.6524353, -104.1531906, 247.3020935, -367.3065796, 388.8056030
2: -101.7458801, 313.4589539, -88.3960266, 272.3433533, -374.0892334, 401.8549805
3: -106.7522278, 407.6034241, -92.7213516, 354.0706177, -460.8227539, 500.3247681
4: -91.1346741, 370.4225769, -79.3333664, 321.6880798, -412.8227234, 449.7559509

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B1_B1_A2_A1

### Relational analysis result of IS_B2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7560124, upper bound: 339.7891453
time: 1.16 seconds

## Relational analysis of IS_B2_B1_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B1_B1_A2_B1

### Relational analysis result of IS_B2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7637479, upper bound: 339.7899493
time: 1.04 seconds

## Relational analysis of IS_B2_B1_B1_B1_A2_B2

### Relational analysis result of IS_B2_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7637484, upper bound: 339.7899006
time: 1.23 seconds

## BFS IS instance: IS_B2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -74.6870956, 250.2789154, -335.3739014, 358.8575745
1: -119.3920059, 282.0884705, -104.7031937, 248.7222290, -368.1141357, 386.7916565
2: -101.2751236, 310.7092590, -88.7651215, 274.1075439, -375.3825684, 399.4743652
3: -106.2089005, 403.8296509, -93.2395935, 356.0269470, -462.2358398, 497.0692444
4: -90.6926956, 367.2429504, -79.6211777, 323.7671204, -414.4597473, 446.8641357

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B1_B2_A1_A1

### Relational analysis result of IS_B2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7551124, upper bound: 339.7835398
time: 0.84 seconds

## Relational analysis of IS_B2_B1_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B1_B2_A1_A1

### Relational analysis result of IS_B2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7522790, upper bound: 339.7795640
time: 0.97 seconds

## Relational analysis of IS_B2_B1_B1_B2_A1_A2

### Relational analysis result of IS_B2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7575209, upper bound: 339.7801206
time: 1.19 seconds

## BFS IS instance: IS_B2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -85.5067825, 287.2420959, -74.6870956, 250.2789154, -335.7857056, 361.9291687
1: -120.1114273, 284.9082947, -104.7031937, 248.7222290, -368.8335266, 389.6114807
2: -101.8361511, 313.7402954, -88.7651215, 274.1075439, -375.9436340, 402.5054321
3: -106.8467102, 407.9730225, -93.2395935, 356.0269470, -462.8736572, 501.2126160
4: -91.2148666, 370.7578735, -79.6211777, 323.7671204, -414.9819641, 450.3790588

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B1_B2_A2_A1

### Relational analysis result of IS_B2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7515413, upper bound: 339.7802862
time: 1.16 seconds

## Relational analysis of IS_B2_B1_B1_B2_A2_A2

### Relational analysis result of IS_B2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7575209, upper bound: 339.7802111
time: 1.04 seconds

## BFS IS instance: IS_B2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -85.0230331, 283.9196167, -82.2580032, 273.6946106, -358.7176514, 366.1776123
1: -119.2899094, 281.8424988, -115.2041168, 271.6928101, -390.9827271, 397.0466309
2: -101.1890335, 310.4364929, -97.7027740, 299.2369080, -400.4259338, 408.1392517
3: -106.1187134, 403.4731140, -102.4676895, 388.6105042, -494.7292175, 505.9407959
4: -90.6161499, 366.9203796, -87.4080505, 353.4977722, -444.1139221, 454.3283691

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B2_B1_A1_A1

### Relational analysis result of IS_B2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7554396, upper bound: 339.7852279
time: 1.19 seconds

## Relational analysis of IS_B2_B1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B1_A1_B1

### Relational analysis result of IS_B2_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7592832, upper bound: 339.7892126
time: 1.09 seconds

## Relational analysis of IS_B2_B1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_B1_B2_B1_A1_A1

### Relational analysis result of IS_B2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7644863, upper bound: 339.7900043
time: 1.10 seconds

## Relational analysis of IS_B2_B1_B2_B1_A1_A2

### Relational analysis result of IS_B2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7644863, upper bound: 339.7900043
time: 1.22 seconds

## BFS IS instance: IS_B2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -85.4314194, 286.9815979, -82.2580032, 273.6946106, -359.1260376, 369.2395935
1: -120.0044708, 284.6524353, -115.2041168, 271.6928101, -391.6972656, 399.8565674
2: -101.7458801, 313.4589539, -97.7027740, 299.2369080, -400.9827881, 411.1617126
3: -106.7522278, 407.6034241, -102.4676895, 388.6105042, -495.3626709, 510.0711060
4: -91.1346741, 370.4225769, -87.4080505, 353.4977722, -444.6324463, 457.8305969

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B2_B1_A2_A1

### Relational analysis result of IS_B2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7554396, upper bound: 339.7853977
time: 1.07 seconds

## Relational analysis of IS_B2_B1_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B1_A2_B1

### Relational analysis result of IS_B2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7592832, upper bound: 339.7893030
time: 1.02 seconds

## Relational analysis of IS_B2_B1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_B1_B2_B1_A2_A1

### Relational analysis result of IS_B2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7644863, upper bound: 339.7901190
time: 1.31 seconds

## Relational analysis of IS_B2_B1_B2_B1_A2_A2

### Relational analysis result of IS_B2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7644863, upper bound: 339.7901190
time: 0.96 seconds

## BFS IS instance: IS_B2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -81.8358765, 273.4550781, -358.5500488, 366.0062866
1: -119.3920059, 282.0884705, -114.8749313, 271.7871704, -391.1791687, 396.9634094
2: -101.2751236, 310.7092590, -97.3516312, 299.6034546, -400.8785706, 408.0608826
3: -106.2089005, 403.8296509, -102.2249146, 388.7427979, -494.9516907, 506.0545654
4: -90.6926956, 367.2429504, -87.1974792, 353.8367920, -444.5294800, 454.4404297

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B2_A1_A1

### Relational analysis result of IS_B2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7522136, upper bound: 339.7784861
time: 0.96 seconds

## Relational analysis of IS_B2_B1_B2_B2_A1_A2

### Relational analysis result of IS_B2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7574554, upper bound: 339.7790428
time: 1.20 seconds

## BFS IS instance: IS_B2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -85.5067825, 287.2420959, -81.8358765, 273.4550781, -358.9618530, 369.0778809
1: -120.1114273, 284.9082947, -114.8749313, 271.7871704, -391.8985596, 399.7832031
2: -101.8361511, 313.7402954, -97.3516312, 299.6034546, -401.4396057, 411.0919189
3: -106.8467102, 407.9730225, -102.2249146, 388.7427979, -495.5895081, 510.1979370
4: -91.2148666, 370.7578735, -87.1974792, 353.8367920, -445.0516663, 457.9552917

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B2_A2_A1

### Relational analysis result of IS_B2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7522136, upper bound: 339.7792083
time: 1.07 seconds

## Relational analysis of IS_B2_B1_B2_B2_A2_A2

### Relational analysis result of IS_B2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7574554, upper bound: 339.7791332
time: 1.12 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.70 seconds
IS_B1_A1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7914241, upper bound: 339.7597075
IS_B1_A1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7789483, upper bound: 339.7567831
IS_B1_A1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7887473, upper bound: 339.7560124
IS_B1_A1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7869310, upper bound: 339.7519940
IS_B1_A1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7835398, upper bound: 339.7551124
IS_B1_A1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7832480, upper bound: 339.7513118
IS_B1_A1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7835398, upper bound: 339.7551124
IS_B1_A1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7832480, upper bound: 339.7513118
IS_B1_A1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7899950, upper bound: 339.7649837
IS_B1_A1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7899950, upper bound: 339.7649837
IS_B1_A1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7796715, upper bound: 339.7547384
IS_B1_A1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7802282, upper bound: 339.7599802
IS_B1_A1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7771203, upper bound: 339.7494001
IS_B1_A1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7849361, upper bound: 339.7516390
IS_B1_A1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7784861, upper bound: 339.7522136
IS_B1_A1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7790428, upper bound: 339.7574554
IS_B2_B1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7637479, upper bound: 339.7898049
IS_B2_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7637484, upper bound: 339.7897849
IS_B2_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7637479, upper bound: 339.7899493
IS_B2_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7637484, upper bound: 339.7899006
IS_B2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7522790, upper bound: 339.7795640
IS_B2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7575209, upper bound: 339.7801206
IS_B2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7515413, upper bound: 339.7802862
IS_B2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7575209, upper bound: 339.7802111
IS_B2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7644863, upper bound: 339.7900043
IS_B2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7644863, upper bound: 339.7900043
IS_B2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7644863, upper bound: 339.7901190
IS_B2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7644863, upper bound: 339.7901190
IS_B2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7522136, upper bound: 339.7784861
IS_B2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7574554, upper bound: 339.7790428
IS_B2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7522136, upper bound: 339.7792083
IS_B2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.70
Output dim: 0, lower bound: -339.7574554, upper bound: 339.7791332

## BFS IS instance: IS_B1_A1_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -66.6641235, 221.5711365, -85.0230331, 283.9196167, -350.5837402, 306.5941162
1: -93.1021500, 220.1848602, -119.2899094, 281.8424988, -374.9446411, 339.4747620
2: -79.0303879, 242.5976868, -101.1890335, 310.4364929, -389.4668884, 343.7867126
3: -82.9030151, 315.0477905, -106.1187134, 403.4731140, -486.3761292, 421.1665039
4: -70.8799667, 286.5773621, -90.6161499, 366.9203796, -437.8003540, 377.1935120

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B1_A1_A1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B1_A1_A1_A1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7833514, upper bound: 339.7343756
time: 1.01 seconds

## Relational analysis of IS_B1_A1_A1_A1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7920274, upper bound: 339.7608192
time: 0.86 seconds

## BFS IS instance: IS_B1_A1_A1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -69.9503403, 232.8013306, -85.0230331, 283.9196167, -353.8699646, 317.8242798
1: -97.3283615, 231.1916046, -119.2899094, 281.8424988, -379.1708679, 350.4815063
2: -82.6892624, 254.7339325, -101.1890335, 310.4364929, -393.1257629, 355.9229736
3: -86.7036285, 330.9612427, -106.1187134, 403.4731140, -490.1767578, 437.0799561
4: -74.3581390, 300.8416443, -90.6161499, 366.9203796, -441.2785034, 391.4577942

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_A1_A1_A1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7803699, upper bound: 339.7581628
time: 1.10 seconds

## Relational analysis of IS_B1_A1_A1_A1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7803699, upper bound: 339.7581628
time: 1.40 seconds

## BFS IS instance: IS_B1_A1_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -74.5175018, 249.2518616, -76.4543381, 257.7856140, -332.3030396, 325.7062073
1: -104.1531906, 247.3020935, -107.1652908, 255.6117249, -359.7649231, 354.4673767
2: -88.3960266, 272.3433533, -90.8572006, 281.6195984, -370.0156250, 363.2005615
3: -92.7213516, 354.0706177, -95.4508438, 366.3883972, -459.1097412, 449.5213928
4: -79.3333664, 321.6880798, -81.5285416, 333.0022583, -412.3356323, 403.2166138

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B1_A1_A1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7862587, upper bound: 339.7537767
time: 0.94 seconds

## Relational analysis of IS_B1_A1_A1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7857820, upper bound: 339.7537483
time: 1.10 seconds

## BFS IS instance: IS_B1_A1_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -74.5175018, 249.2518616, -84.1508942, 280.7848816, -355.3023682, 333.4027710
1: -104.1531906, 247.3020935, -118.0812225, 278.7597656, -382.9129639, 365.3833008
2: -88.3960266, 272.3433533, -100.1760635, 307.0471191, -395.4431458, 372.5194092
3: -92.7213516, 354.0706177, -105.0398636, 399.0413818, -491.7626953, 459.1104126
4: -79.3333664, 321.6880798, -89.7115021, 362.9685364, -442.3019104, 411.3995972

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B1_A1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B1_A1_A1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7844676, upper bound: 339.7499406
time: 1.17 seconds

## Relational analysis of IS_B1_A1_A1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7847939, upper bound: 339.7501127
time: 0.92 seconds

## BFS IS instance: IS_B1_A1_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -73.8942566, 245.9733887, -76.5251465, 258.0331726, -331.9273682, 322.4985352
1: -103.4220047, 244.7231598, -107.2658615, 255.8545990, -359.2766113, 351.9890137
2: -87.7495346, 269.7869873, -90.9418640, 281.8870544, -369.6365967, 360.7288513
3: -92.1146774, 350.1335144, -95.5396729, 366.7406006, -458.8552856, 445.6731873
4: -78.7043076, 318.6690674, -81.6038895, 333.3200378, -412.0243530, 400.2729492

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_A1_A2_A1_B1_B1

### Relational analysis result of IS_B1_A1_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7784569, upper bound: 339.7413226
time: 1.11 seconds

## Relational analysis of IS_B1_A1_A1_A2_A1_B1_B2

### Relational analysis result of IS_B1_A1_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7790543, upper bound: 339.7477614
time: 1.03 seconds

## BFS IS instance: IS_B1_A1_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -73.8942566, 245.9733887, -84.2214584, 281.0291443, -354.9234009, 330.1948547
1: -103.4220047, 244.7231598, -118.1812286, 278.9996948, -382.4216309, 362.9043884
2: -87.7495346, 269.7869873, -100.2603912, 307.3133545, -395.0628967, 370.0473328
3: -92.1146774, 350.1335144, -105.1281967, 399.3886414, -491.5033264, 455.2617188
4: -78.7043076, 318.6690674, -89.7864609, 363.2832031, -441.9875183, 408.4555054

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B1_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B1_A1_A1_A2_A1_B2_A1

### Relational analysis result of IS_B1_A1_A1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7842646, upper bound: 339.7522889
time: 1.18 seconds

## Relational analysis of IS_B1_A1_A1_A2_A1_B2_A2

### Relational analysis result of IS_B1_A1_A1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7856971, upper bound: 339.7521956
time: 0.90 seconds

## BFS IS instance: IS_B1_A1_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -74.6870956, 250.2789154, -76.5251465, 258.0331726, -332.7202454, 326.8040771
1: -104.7031937, 248.7222290, -107.2658615, 255.8545990, -360.5578003, 355.9879761
2: -88.7651215, 274.1075439, -90.9418640, 281.8870544, -370.6521606, 365.0493774
3: -93.2395935, 356.0269470, -95.5396729, 366.7406006, -459.9801941, 451.5666199
4: -79.6211777, 323.7671204, -81.6038895, 333.3200378, -412.9412231, 405.3709717

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_A1_A2_A2_B1_B1

### Relational analysis result of IS_B1_A1_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7752289, upper bound: 339.7372369
time: 0.89 seconds

## Relational analysis of IS_B1_A1_A1_A2_A2_B1_B2

### Relational analysis result of IS_B1_A1_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7758264, upper bound: 339.7436758
time: 0.79 seconds

## BFS IS instance: IS_B1_A1_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -74.6870956, 250.2789154, -84.2214584, 281.0291443, -355.7162476, 334.5003662
1: -104.7031937, 248.7222290, -118.1812286, 278.9996948, -383.7028809, 366.9033508
2: -88.7651215, 274.1075439, -100.2603912, 307.3133545, -396.0784912, 374.3677979
3: -93.2395935, 356.0269470, -105.1281967, 399.3886414, -492.6282349, 461.1551208
4: -79.6211777, 323.7671204, -89.7864609, 363.2832031, -442.9043884, 413.5535278

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B1_A1_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B1_A1_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_A1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A1_A1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7812569, upper bound: 339.7427446
time: 0.84 seconds

## Relational analysis of IS_B1_A1_A1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A1_A1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7755617, upper bound: 339.7412380
time: 0.99 seconds

## BFS IS instance: IS_B1_A1_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -82.5946960, 273.2151489, -82.9117126, 276.7777710, -359.3724670, 356.1268616
1: -115.5130005, 271.4657898, -116.3173447, 274.8097229, -390.3226624, 387.7831116
2: -98.0321426, 299.0640259, -98.7043610, 302.7190552, -400.7511902, 397.7683105
3: -102.7604294, 387.9916077, -103.4839859, 393.2551880, -496.0156250, 491.4755859
4: -87.6726685, 353.1832581, -88.4066391, 357.5976868, -445.2703552, 441.5899048

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_A2_A1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7892369, upper bound: 339.7598044
time: 0.97 seconds

## Relational analysis of IS_B1_A1_A2_A1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_A2_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7862848, upper bound: 339.7562695
time: 1.01 seconds

## Relational analysis of IS_B1_A1_A2_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7859931, upper bound: 339.7524689
time: 1.04 seconds

## BFS IS instance: IS_B1_A1_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -82.5946960, 273.2151489, -97.0845490, 328.8274536, -411.4221497, 370.2996826
1: -115.5130005, 271.4657898, -136.4039764, 325.8498840, -441.3628540, 407.8697510
2: -98.0321426, 299.0640259, -115.7457733, 358.7676697, -456.7998047, 414.8097839
3: -102.7604294, 387.9916077, -121.3922348, 466.0828552, -568.8432617, 509.3838501
4: -87.6726685, 353.1832581, -103.6152573, 423.2085876, -510.8812561, 456.7984619

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_A2_A1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7892369, upper bound: 339.7598044
time: 0.85 seconds

## Relational analysis of IS_B1_A1_A2_A1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_A2_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7862848, upper bound: 339.7562695
time: 1.02 seconds

## Relational analysis of IS_B1_A1_A2_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7859931, upper bound: 339.7524689
time: 1.21 seconds

## BFS IS instance: IS_B1_A1_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -81.1868820, 269.6222839, -77.4703369, 258.9702148, -340.1570435, 347.0926208
1: -113.7671432, 268.2580566, -108.7526398, 257.1856079, -370.9527588, 377.0106812
2: -96.4886932, 295.8059998, -92.2024002, 283.3155212, -379.8041992, 388.0083923
3: -101.2618713, 383.4673462, -96.7372665, 368.1341858, -469.3960571, 480.2045288
4: -86.4228897, 349.2984009, -82.6163330, 334.8500671, -421.2729492, 431.9147339

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_A2_A1_A2_B1_B1

### Relational analysis result of IS_B1_A1_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7767214, upper bound: 339.7407409
time: 1.03 seconds

## Relational analysis of IS_B1_A1_A2_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_A1_A2_A1_A2_B1_B1

### Relational analysis result of IS_B1_A1_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7796715, upper bound: 339.7547384
time: 1.12 seconds

## Relational analysis of IS_B1_A1_A2_A1_A2_B1_B2

### Relational analysis result of IS_B1_A1_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7796715, upper bound: 339.7547384
time: 0.97 seconds

## BFS IS instance: IS_B1_A1_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -81.1868820, 269.6222839, -80.7415924, 270.3125000, -351.4992981, 350.3638916
1: -113.7671432, 268.2580566, -113.1016769, 268.2474976, -382.0146179, 381.3597412
2: -96.4886932, 295.8059998, -95.9243317, 295.5010071, -391.9896851, 391.7302856
3: -101.2618713, 383.4673462, -100.6454315, 384.2853699, -485.5472412, 484.1127930
4: -86.4228897, 349.2984009, -85.9844360, 349.3599243, -435.7828064, 435.2828369

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_A2_A1_A2_B2_B1

### Relational analysis result of IS_B1_A1_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7773189, upper bound: 339.7471798
time: 1.33 seconds

## Relational analysis of IS_B1_A1_A2_A1_A2_B2_B2

### Relational analysis result of IS_B1_A1_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7770542, upper bound: 339.7447420
time: 1.07 seconds

## BFS IS instance: IS_B1_A1_A2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -75.6055756, 253.7299500, -85.0230331, 283.9196167, -359.5251465, 338.7529907
1: -105.7764359, 251.6033630, -119.2899094, 281.8424988, -387.6189270, 370.8932800
2: -89.6408310, 277.2070312, -101.1890335, 310.4364929, -400.0773010, 378.3960571
3: -94.1737976, 360.4662170, -106.1187134, 403.4731140, -497.6469116, 466.5849304
4: -80.3808365, 327.8102722, -90.6161499, 366.9203796, -447.3012085, 418.4264221

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B1_A1_A2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B1_A1_A2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B1_A1_A2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_A1_A2_A2_A1_A1_B1

### Relational analysis result of IS_B1_A1_A2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7771203, upper bound: 339.7494001
time: 0.92 seconds

## Relational analysis of IS_B1_A1_A2_A2_A1_A1_B2

### Relational analysis result of IS_B1_A1_A2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7771203, upper bound: 339.7494001
time: 1.42 seconds

## BFS IS instance: IS_B1_A1_A2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -81.0973663, 269.7848816, -85.0230331, 283.9196167, -365.0169678, 354.8079224
1: -113.6655426, 267.8550415, -119.2899094, 281.8424988, -395.5080566, 387.1448975
2: -96.3990097, 295.0073242, -101.1890335, 310.4364929, -406.8355103, 396.1963501
3: -101.0933151, 383.0970154, -106.1187134, 403.4731140, -504.5664062, 489.2157288
4: -86.2322235, 348.5137329, -90.6161499, 366.9203796, -453.1525879, 439.1298828

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_A2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B1_A1_A2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B1_A1_A2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_A2_A2_A1_A2_A1

### Relational analysis result of IS_B1_A1_A2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7834810, upper bound: 339.7431441
time: 1.01 seconds

## Relational analysis of IS_B1_A1_A2_A2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B1_A1_A2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_A1_A2_A2_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7849361, upper bound: 339.7516390
time: 0.87 seconds

## Relational analysis of IS_B1_A1_A2_A2_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7849361, upper bound: 339.7516390
time: 1.23 seconds

## BFS IS instance: IS_B1_A1_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -81.8358765, 273.4550781, -77.4703369, 258.9702148, -340.8060303, 350.9254150
1: -114.8749313, 271.7871704, -108.7526398, 257.1856079, -372.0605469, 380.5397949
2: -97.3516312, 299.6034546, -92.2024002, 283.3155212, -380.6671448, 391.8058472
3: -102.2249146, 388.7427979, -96.7372665, 368.1341858, -470.3591003, 485.4799805
4: -87.1974792, 353.8367920, -82.6163330, 334.8500671, -422.0475159, 436.4531250

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_A2_A2_A2_B1_B1

### Relational analysis result of IS_B1_A1_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7729162, upper bound: 339.7370903
time: 1.07 seconds

## Relational analysis of IS_B1_A1_A2_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B1_A1_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_A1_A2_A2_A2_B1_B1

### Relational analysis result of IS_B1_A1_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7784861, upper bound: 339.7522136
time: 0.97 seconds

## Relational analysis of IS_B1_A1_A2_A2_A2_B1_B2

### Relational analysis result of IS_B1_A1_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7784861, upper bound: 339.7522136
time: 1.18 seconds

## BFS IS instance: IS_B1_A1_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -81.8358765, 273.4550781, -80.7415924, 270.3125000, -352.1482849, 354.1966553
1: -114.8749313, 271.7871704, -113.1016769, 268.2474976, -383.1224060, 384.8888550
2: -97.3516312, 299.6034546, -95.9243317, 295.5010071, -392.8526306, 395.5277710
3: -102.2249146, 388.7427979, -100.6454315, 384.2853699, -486.5102844, 489.3882446
4: -87.1974792, 353.8367920, -85.9844360, 349.3599243, -436.5573730, 439.8211975

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_A2_A2_A2_B2_B1

### Relational analysis result of IS_B1_A1_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7735136, upper bound: 339.7435291
time: 1.49 seconds

## Relational analysis of IS_B1_A1_A2_A2_A2_B2_B2

### Relational analysis result of IS_B1_A1_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7732490, upper bound: 339.7410913
time: 0.89 seconds

## BFS IS instance: IS_B2_B1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -84.9825668, 283.7807617, -72.6436005, 242.5830078, -327.5655212, 356.4243774
1: -119.2330933, 281.7062988, -101.4604797, 241.0747833, -360.3077698, 383.1667786
2: -101.1413956, 310.2866821, -86.1406174, 265.6739807, -366.8153687, 396.4273071
3: -106.0683365, 403.2768250, -90.3380737, 345.2267761, -451.2950745, 493.6148987
4: -90.5739212, 366.7424927, -77.3927765, 313.9828491, -404.5567627, 444.1352539

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B1_B1_A1_B1_A1

### Relational analysis result of IS_B2_B1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7537767, upper bound: 339.7862587
time: 1.02 seconds

## Relational analysis of IS_B2_B1_B1_B1_A1_B1_A2

### Relational analysis result of IS_B2_B1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7499407, upper bound: 339.7844676
time: 1.08 seconds

## BFS IS instance: IS_B2_B1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -85.0230331, 283.9196167, -71.9332123, 240.3917542, -325.4147644, 355.8528137
1: -119.2899094, 281.8424988, -100.3580170, 238.5715637, -357.8614502, 382.2005005
2: -101.1890335, 310.4364929, -85.2100449, 262.8039551, -363.9929810, 395.6465149
3: -106.1187134, 403.4731140, -89.3832245, 341.5888977, -447.7076111, 492.8563232
4: -90.6161499, 366.9203796, -76.5594788, 310.3712463, -400.9873962, 443.4798279

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B1_B1_A1_B2_A1

### Relational analysis result of IS_B2_B1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7537483, upper bound: 339.7857820
time: 0.93 seconds

## Relational analysis of IS_B2_B1_B1_B1_A1_B2_A2

### Relational analysis result of IS_B2_B1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7501127, upper bound: 339.7847939
time: 1.43 seconds

## BFS IS instance: IS_B2_B1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -85.3881454, 286.8334351, -72.6436005, 242.5830078, -327.9711609, 359.4770508
1: -119.9439011, 284.5067444, -101.4604797, 241.0747833, -361.0186462, 385.9672241
2: -101.6950989, 313.2989197, -86.1406174, 265.6739807, -367.3690796, 399.4395447
3: -106.6984558, 407.3940125, -90.3380737, 345.2267761, -451.9252319, 497.7320862
4: -91.0896149, 370.2327576, -77.3927765, 313.9828491, -405.0724487, 447.6254883

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B1_B1_A2_B1_A1

### Relational analysis result of IS_B2_B1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7559687, upper bound: 339.7868085
time: 0.94 seconds

## Relational analysis of IS_B2_B1_B1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B1_B1_A2_B1_B1

### Relational analysis result of IS_B2_B1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7591361, upper bound: 339.7891987
time: 1.09 seconds

## Relational analysis of IS_B2_B1_B1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_B1_B1_B1_A2_B1_A1

### Relational analysis result of IS_B2_B1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7655850, upper bound: 339.7899493
time: 1.03 seconds

## Relational analysis of IS_B2_B1_B1_B1_A2_B1_A2

### Relational analysis result of IS_B2_B1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7655850, upper bound: 339.7899493
time: 0.83 seconds

## BFS IS instance: IS_B2_B1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -85.4314194, 286.9815979, -71.9332123, 240.3917542, -325.8231201, 358.9147949
1: -120.0044708, 284.6524353, -100.3580170, 238.5715637, -358.5760498, 385.0104370
2: -101.7458801, 313.4589539, -85.2100449, 262.8039551, -364.5498352, 398.6689758
3: -106.7522278, 407.6034241, -89.3832245, 341.5888977, -448.3410645, 496.9866333
4: -91.1346741, 370.4225769, -76.5594788, 310.3712463, -401.5059204, 446.9820557

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B1_B1_A2_B2_A1

### Relational analysis result of IS_B2_B1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7558590, upper bound: 339.7859821
time: 1.20 seconds

## Relational analysis of IS_B2_B1_B1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B1_B1_A2_B2_B1

### Relational analysis result of IS_B2_B1_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7591219, upper bound: 339.7891497
time: 0.91 seconds

## Relational analysis of IS_B2_B1_B1_B1_A2_B2_B2

### Relational analysis result of IS_B2_B1_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7561973, upper bound: 339.7763772
time: 0.87 seconds

## BFS IS instance: IS_B2_B1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -77.4703369, 258.9702148, -74.6870956, 250.2789154, -327.7492676, 333.6573181
1: -108.7526398, 257.1856079, -104.7031937, 248.7222290, -357.4747925, 361.8887939
2: -92.2024002, 283.3155212, -88.7651215, 274.1075439, -366.3099365, 372.0806274
3: -96.7372665, 368.1341858, -93.2395935, 356.0269470, -452.7641602, 461.3737793
4: -82.6163330, 334.8500671, -79.6211777, 323.7671204, -406.3834229, 414.4712524

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_A1

### Relational analysis result of IS_B2_B1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7372369, upper bound: 339.7752289
time: 0.97 seconds

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_A1

### Relational analysis result of IS_B2_B1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7522790, upper bound: 339.7795640
time: 1.09 seconds

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_A2

### Relational analysis result of IS_B2_B1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7522790, upper bound: 339.7795640
time: 0.91 seconds

## BFS IS instance: IS_B2_B1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -80.7415924, 270.3125000, -74.6870956, 250.2789154, -331.0205078, 344.9995728
1: -113.1016769, 268.2474976, -104.7031937, 248.7222290, -361.8238220, 372.9506836
2: -95.9243317, 295.5010071, -88.7651215, 274.1075439, -370.0317688, 384.2661133
3: -100.6454315, 384.2853699, -93.2395935, 356.0269470, -456.6723633, 477.5249634
4: -85.9844360, 349.3599243, -79.6211777, 323.7671204, -409.7514954, 428.9811096

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_B1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B1_B2_A1_A2_A1

### Relational analysis result of IS_B2_B1_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7436758, upper bound: 339.7758264
time: 1.37 seconds

## Relational analysis of IS_B2_B1_B1_B2_A1_A2_A2

### Relational analysis result of IS_B2_B1_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7412380, upper bound: 339.7755618
time: 0.92 seconds

## BFS IS instance: IS_B2_B1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -77.7702713, 261.6948853, -74.6870956, 250.2789154, -328.0491943, 336.3819885
1: -109.3317337, 259.6387329, -104.7031937, 248.7222290, -358.0538940, 364.3419189
2: -92.6404495, 285.9417114, -88.7651215, 274.1075439, -366.7478943, 374.7068481
3: -97.2451324, 371.8041382, -93.2395935, 356.0269470, -453.2720642, 465.0437317
4: -83.0186615, 337.8824768, -79.6211777, 323.7671204, -406.7857666, 417.5036621

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_A1

### Relational analysis result of IS_B2_B1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7455989, upper bound: 339.7760074
time: 1.14 seconds

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_A1

### Relational analysis result of IS_B2_B1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7592507, upper bound: 339.7802862
time: 1.04 seconds

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_A2

### Relational analysis result of IS_B2_B1_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7592507, upper bound: 339.7802862
time: 0.96 seconds

## BFS IS instance: IS_B2_B1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -81.2579422, 273.7367249, -74.6870956, 250.2789154, -331.5368652, 348.4238281
1: -113.9449158, 271.4089966, -104.7031937, 248.7222290, -362.6670837, 376.1121826
2: -96.5973206, 298.9041748, -88.7651215, 274.1075439, -370.7048340, 387.6693115
3: -101.3996277, 388.9440002, -93.2395935, 356.0269470, -457.4265747, 482.1835938
4: -86.6249771, 353.3235474, -79.6211777, 323.7671204, -410.3920288, 432.9447021

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_A1

### Relational analysis result of IS_B2_B1_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7446092, upper bound: 339.7759207
time: 1.15 seconds

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_A1

### Relational analysis result of IS_B2_B1_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7584241, upper bound: 339.7802111
time: 1.15 seconds

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_A2

### Relational analysis result of IS_B2_B1_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7584241, upper bound: 339.7802111
time: 1.02 seconds

## BFS IS instance: IS_B2_B1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -82.9117126, 276.7777710, -82.2580032, 273.6946106, -356.6063232, 359.0357666
1: -116.3173447, 274.8097229, -115.2041168, 271.6928101, -388.0101624, 390.0138245
2: -98.7043610, 302.7190552, -97.7027740, 299.2369080, -397.9412842, 400.4218140
3: -103.4839859, 393.2551880, -102.4676895, 388.6105042, -492.0944824, 495.7228699
4: -88.4066391, 357.5976868, -87.4080505, 353.4977722, -441.9044189, 445.0057068

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B2_B1_A1_A1_A1

### Relational analysis result of IS_B2_B1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7554396, upper bound: 339.7852279
time: 1.22 seconds

## Relational analysis of IS_B2_B1_B2_B1_A1_A1_A2

### Relational analysis result of IS_B2_B1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7516390, upper bound: 339.7849361
time: 0.94 seconds

## BFS IS instance: IS_B2_B1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -97.0845490, 328.8274536, -82.2580032, 273.6946106, -370.7791443, 411.0854492
1: -136.4039764, 325.8498840, -115.2041168, 271.6928101, -408.0968018, 441.0539856
2: -115.7457733, 358.7676697, -97.7027740, 299.2369080, -414.9826660, 456.4704285
3: -121.3922348, 466.0828552, -102.4676895, 388.6105042, -510.0027466, 568.5505371
4: -103.6152573, 423.2085876, -87.4080505, 353.4977722, -457.1129761, 510.6166077

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B1_A1_A2_B1

### Relational analysis result of IS_B2_B1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7592832, upper bound: 339.7892126
time: 1.04 seconds

## Relational analysis of IS_B2_B1_B2_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B2_B1_A1_A2_A1

### Relational analysis result of IS_B2_B1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7554396, upper bound: 339.7852279
time: 1.14 seconds

## Relational analysis of IS_B2_B1_B2_B1_A1_A2_A2

### Relational analysis result of IS_B2_B1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7516390, upper bound: 339.7849361
time: 1.21 seconds

## BFS IS instance: IS_B2_B1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -83.4334793, 280.1829834, -82.2580032, 273.6946106, -357.1280823, 362.4409485
1: -117.2164764, 277.9400330, -115.2041168, 271.6928101, -388.9093018, 393.1441650
2: -99.4001694, 306.0918884, -97.7027740, 299.2369080, -398.6370850, 403.7946472
3: -104.2745743, 397.9013672, -102.4676895, 388.6105042, -492.8850708, 500.3690491
4: -89.0465927, 361.6151428, -87.4080505, 353.4977722, -442.5443726, 449.0231628

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_A1

### Relational analysis result of IS_B2_B1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7576586, upper bound: 339.7853977
time: 0.90 seconds

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_B1

### Relational analysis result of IS_B2_B1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7601864, upper bound: 339.7893030
time: 1.06 seconds

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_B1

### Relational analysis result of IS_B2_B1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7661029, upper bound: 339.7900602
time: 0.83 seconds

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_B2

### Relational analysis result of IS_B2_B1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7662410, upper bound: 339.7900620
time: 1.44 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 14.21 seconds
IS_B1_A1_A1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7833514, upper bound: 339.7343756
IS_B1_A1_A1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7920274, upper bound: 339.7608192
IS_B1_A1_A1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7803699, upper bound: 339.7581628
IS_B1_A1_A1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7803699, upper bound: 339.7581628
IS_B1_A1_A1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7862587, upper bound: 339.7537767
IS_B1_A1_A1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7857820, upper bound: 339.7537483
IS_B1_A1_A1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7844676, upper bound: 339.7499406
IS_B1_A1_A1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7847939, upper bound: 339.7501127
IS_B1_A1_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7784569, upper bound: 339.7413226
IS_B1_A1_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7790543, upper bound: 339.7477614
IS_B1_A1_A1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7842646, upper bound: 339.7522889
IS_B1_A1_A1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7856971, upper bound: 339.7521956
IS_B1_A1_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7752289, upper bound: 339.7372369
IS_B1_A1_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7758264, upper bound: 339.7436758
IS_B1_A1_A1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7812569, upper bound: 339.7427446
IS_B1_A1_A1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7755617, upper bound: 339.7412380
IS_B1_A1_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7862848, upper bound: 339.7562695
IS_B1_A1_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7859931, upper bound: 339.7524689
IS_B1_A1_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7862848, upper bound: 339.7562695
IS_B1_A1_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7859931, upper bound: 339.7524689
IS_B1_A1_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7796715, upper bound: 339.7547384
IS_B1_A1_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7796715, upper bound: 339.7547384
IS_B1_A1_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7773189, upper bound: 339.7471798
IS_B1_A1_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7770542, upper bound: 339.7447420
IS_B1_A1_A2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7771203, upper bound: 339.7494001
IS_B1_A1_A2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7771203, upper bound: 339.7494001
IS_B1_A1_A2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7849361, upper bound: 339.7516390
IS_B1_A1_A2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7849361, upper bound: 339.7516390
IS_B1_A1_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7784861, upper bound: 339.7522136
IS_B1_A1_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7784861, upper bound: 339.7522136
IS_B1_A1_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7735136, upper bound: 339.7435291
IS_B1_A1_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7732490, upper bound: 339.7410913
IS_B2_B1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7537767, upper bound: 339.7862587
IS_B2_B1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7499407, upper bound: 339.7844676
IS_B2_B1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7537483, upper bound: 339.7857820
IS_B2_B1_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7501127, upper bound: 339.7847939
IS_B2_B1_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7655850, upper bound: 339.7899493
IS_B2_B1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7655850, upper bound: 339.7899493
IS_B2_B1_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7591219, upper bound: 339.7891497
IS_B2_B1_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7561973, upper bound: 339.7763772
IS_B2_B1_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7522790, upper bound: 339.7795640
IS_B2_B1_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7522790, upper bound: 339.7795640
IS_B2_B1_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7436758, upper bound: 339.7758264
IS_B2_B1_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7412380, upper bound: 339.7755618
IS_B2_B1_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7592507, upper bound: 339.7802862
IS_B2_B1_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7592507, upper bound: 339.7802862
IS_B2_B1_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7584241, upper bound: 339.7802111
IS_B2_B1_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7584241, upper bound: 339.7802111
IS_B2_B1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7554396, upper bound: 339.7852279
IS_B2_B1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7516390, upper bound: 339.7849361
IS_B2_B1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7554396, upper bound: 339.7852279
IS_B2_B1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7516390, upper bound: 339.7849361
IS_B2_B1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7661029, upper bound: 339.7900602
IS_B2_B1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 14.21
Output dim: 0, lower bound: -339.7662410, upper bound: 339.7900620
IS_B2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.21
Output dim: 0, lower bound: -339.7644863, upper bound: 339.7901190
IS_B2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 14.21
Output dim: 0, lower bound: -339.7522136, upper bound: 339.7784861
IS_B2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 14.21
Output dim: 0, lower bound: -339.7574554, upper bound: 339.7790428
IS_B2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.21
Output dim: 0, lower bound: -339.7522136, upper bound: 339.7792083
IS_B2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.21
Output dim: 0, lower bound: -339.7574554, upper bound: 339.7791332
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=385.80084228515625
rel_dist={0: [-339.80568762378186, 339.805687623782]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8053471, upper bound: 339.8042640
time: 1.26 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042151, upper bound: 339.8042151
time: 1.12 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.59 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 2.59
Output dim: 0, lower bound: -339.8053471, upper bound: 339.8042640
IS_B2, status: Status.UNKNOWN, split count: 1, time: 2.59
Output dim: 0, lower bound: -339.8042151, upper bound: 339.8042151

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -88.4568100, 296.2978516, -85.0949707, 284.1704712, -372.6272888, 381.3928223
1: -124.1144333, 294.0228577, -119.3920059, 282.0884705, -406.2029114, 413.4148254
2: -105.2662048, 323.7992249, -101.2751236, 310.7092590, -415.9754028, 425.0743408
3: -110.4195175, 420.8130493, -106.2089005, 403.8296509, -514.2490845, 527.0219727
4: -94.2557297, 382.5349121, -90.6926956, 367.2429504, -461.4986877, 473.2276001

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042151, upper bound: 339.8042151
time: 1.35 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042151, upper bound: 339.8042151
time: 1.10 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -88.6961975, 297.1046753, -85.5067825, 287.2420959, -375.9382324, 382.6114502
1: -124.4471970, 294.8176575, -120.1114273, 284.9082947, -409.3554993, 414.9290466
2: -105.5478058, 324.6724243, -101.8361511, 313.7402954, -419.2880859, 426.5085449
3: -110.7164154, 421.9519958, -106.8467102, 407.9730225, -518.6894531, 528.7987061
4: -94.5076294, 383.5692749, -91.2148666, 370.7578735, -465.2655029, 474.7841492

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042151, upper bound: 339.8042151
time: 1.14 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042151, upper bound: 339.8042151
time: 1.25 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.86 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 4.86
Output dim: 0, lower bound: -339.8042151, upper bound: 339.8042151
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 4.86
Output dim: 0, lower bound: -339.8042151, upper bound: 339.8042151
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 4.86
Output dim: 0, lower bound: -339.8042151, upper bound: 339.8042151
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 4.86
Output dim: 0, lower bound: -339.8042151, upper bound: 339.8042151

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -85.0949707, 284.1704712, -369.2654419, 369.2654114
1: -119.3920059, 282.0884705, -119.3920059, 282.0884705, -401.4804077, 401.4804077
2: -101.2751236, 310.7092590, -101.2751236, 310.7092590, -411.9843445, 411.9843445
3: -106.2089005, 403.8296509, -106.2089005, 403.8296509, -510.0385437, 510.0385437
4: -90.6926956, 367.2429504, -90.6926956, 367.2429504, -457.9356384, 457.9356384

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7901989, upper bound: 339.7677381
time: 0.96 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7651631
time: 1.31 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -85.5067825, 287.2420959, -85.0949707, 284.1704712, -369.6772461, 372.3370361
1: -120.1114273, 284.9082947, -119.3920059, 282.0884705, -402.1998596, 404.3002319
2: -101.8361511, 313.7402954, -101.2751236, 310.7092590, -412.5454102, 415.0153809
3: -106.8467102, 407.9730225, -106.2089005, 403.8296509, -510.6763611, 514.1819458
4: -91.2148666, 370.7578735, -90.6926956, 367.2429504, -458.4578247, 461.4505310

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7901989, upper bound: 339.7677381
time: 1.06 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7653524
time: 1.25 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -85.5067825, 287.2420959, -372.3370361, 369.6772461
1: -119.3920059, 282.0884705, -120.1114273, 284.9082947, -404.3002319, 402.1998596
2: -101.2751236, 310.7092590, -101.8361511, 313.7402954, -415.0153809, 412.5454102
3: -106.2089005, 403.8296509, -106.8467102, 407.9730225, -514.1819458, 510.6763611
4: -90.6926956, 367.2429504, -91.2148666, 370.7578735, -461.4505615, 458.4578247

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7677381, upper bound: 339.7901989
time: 0.94 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7672412
time: 1.03 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -85.5067825, 287.2420959, -85.5067825, 287.2420959, -372.7488708, 372.7488708
1: -120.1114273, 284.9082947, -120.1114273, 284.9082947, -405.0196838, 405.0196838
2: -101.8361511, 313.7402954, -101.8361511, 313.7402954, -415.5764465, 415.5764465
3: -106.8467102, 407.9730225, -106.8467102, 407.9730225, -514.8197021, 514.8197021
4: -91.2148666, 370.7578735, -91.2148666, 370.7578735, -461.9727173, 461.9727173

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7677381, upper bound: 339.7915765
time: 0.94 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7674305
time: 0.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.24 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 0, lower bound: -339.7901989, upper bound: 339.7677381
IS_B1_A1_A2, status: Status.VERIFIED, split count: 3, time: 4.24
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7651631
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 0, lower bound: -339.7901989, upper bound: 339.7677381
IS_B1_A2_A2, status: Status.VERIFIED, split count: 3, time: 4.24
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7653524
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 0, lower bound: -339.7677381, upper bound: 339.7901989
IS_B2_A1_B2, status: Status.VERIFIED, split count: 3, time: 4.24
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7672412
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 0, lower bound: -339.7677381, upper bound: 339.7915765
IS_B2_A2_B2, status: Status.VERIFIED, split count: 3, time: 4.24
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7674305

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -82.9748306, 276.9905396, -85.0949707, 284.1704712, -367.1452637, 362.0854797
1: -116.4064331, 275.0193787, -119.3920059, 282.0884705, -398.4949036, 394.4113464
2: -98.7795639, 302.9554138, -101.2751236, 310.7092590, -409.4888000, 404.2305298
3: -103.5627213, 393.5593262, -106.2089005, 403.8296509, -507.3923645, 499.7682190
4: -88.4737244, 357.8739319, -90.6926956, 367.2429504, -455.7166443, 448.5666199

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7651631
time: 1.14 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7651631
time: 1.16 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -83.4957504, 280.3916321, -85.0949707, 284.1704712, -367.6661987, 365.4866028
1: -117.3040619, 278.1459656, -119.3920059, 282.0884705, -399.3925171, 397.5379639
2: -99.4742203, 306.3189087, -101.2751236, 310.7092590, -410.1834717, 407.5939941
3: -104.3519974, 398.1978760, -106.2089005, 403.8296509, -508.1816406, 504.4067688
4: -89.1125412, 361.8850098, -90.6926956, 367.2429504, -456.3554382, 452.5776978

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7672412, upper bound: 339.7653524
time: 1.04 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7672412, upper bound: 339.7653524
time: 1.25 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -83.4957504, 280.3916321, -365.4866028, 367.6661987
1: -119.3920059, 282.0884705, -117.3040619, 278.1459656, -397.5379639, 399.3925171
2: -101.2751236, 310.7092590, -99.4742203, 306.3189087, -407.5939941, 410.1834717
3: -106.2089005, 403.8296509, -104.3519974, 398.1978760, -504.4067688, 508.1816406
4: -90.6926956, 367.2429504, -89.1125412, 361.8850098, -452.5776978, 456.3554382

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7653524, upper bound: 339.7672412
time: 0.87 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7653524, upper bound: 339.7672412
time: 0.82 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -85.5067825, 287.2420959, -83.4957504, 280.3916321, -365.8984070, 370.7377625
1: -120.1114273, 284.9082947, -117.3040619, 278.1459656, -398.2573547, 402.2123413
2: -101.8361511, 313.7402954, -99.4742203, 306.3189087, -408.1550598, 413.2145081
3: -106.8467102, 407.9730225, -104.3519974, 398.1978760, -505.0445862, 512.3250122
4: -91.2148666, 370.7578735, -89.1125412, 361.8850098, -453.0998840, 459.8703613

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7674243, upper bound: 339.7674305
time: 0.87 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7674243, upper bound: 339.7674305
time: 1.02 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.25 seconds
IS_B1_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.25
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7651631
IS_B1_A1_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.25
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7651631
IS_B1_A2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.25
Output dim: 0, lower bound: -339.7672412, upper bound: 339.7653524
IS_B1_A2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.25
Output dim: 0, lower bound: -339.7672412, upper bound: 339.7653524
IS_B2_A1_B1_A1, status: Status.VERIFIED, split count: 4, time: 4.25
Output dim: 0, lower bound: -339.7653524, upper bound: 339.7672412
IS_B2_A1_B1_A2, status: Status.VERIFIED, split count: 4, time: 4.25
Output dim: 0, lower bound: -339.7653524, upper bound: 339.7672412
IS_B2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 4.25
Output dim: 0, lower bound: -339.7674243, upper bound: 339.7674305
IS_B2_A2_B1_A2, status: Status.VERIFIED, split count: 4, time: 4.25
Output dim: 0, lower bound: -339.7674243, upper bound: 339.7674305
Binary search (step 1): status=Status.VERIFIED, low=0.2500000, high=0.5000000, mid=0.2500000, abs_max=385.80084228515625
rel_dist={0: [-339.8055350744037, 339.8055350744037]}

## Binary search (step 2) starts
Candidate diff: 0.3750000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042672, upper bound: 339.8055942
time: 0.81 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
time: 1.11 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.13 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.13
Output dim: 0, lower bound: -339.8042672, upper bound: 339.8055942
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.13
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -88.6961975, 297.1046753, -382.1996155, 372.8666687
1: -119.3920059, 282.0884705, -124.4471970, 294.8176575, -414.2096252, 406.5356750
2: -101.2751236, 310.7092590, -105.5478058, 324.6724243, -425.9474792, 416.2570801
3: -106.2089005, 403.8296509, -110.7164154, 421.9519958, -528.1608887, 514.5460205
4: -90.6926956, 367.2429504, -94.5076294, 383.5692749, -474.2619629, 461.7505798

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7683727, upper bound: 339.7935269
time: 1.17 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7653524, upper bound: 339.7672412
time: 1.12 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -85.5067825, 287.2420959, -88.6961975, 297.1046753, -382.6114502, 375.9382324
1: -120.1114273, 284.9082947, -124.4471970, 294.8176575, -414.9290466, 409.3554993
2: -101.8361511, 313.7402954, -105.5478058, 324.6724243, -426.5085449, 419.2880859
3: -106.8467102, 407.9730225, -110.7164154, 421.9519958, -528.7987061, 518.6894531
4: -91.2148666, 370.7578735, -94.5076294, 383.5692749, -474.7841492, 465.2655029

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.13 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.01 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.60 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 0, lower bound: -339.7683727, upper bound: 339.7935269
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 4.60
Output dim: 0, lower bound: -339.7653524, upper bound: 339.7672412
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -86.7272873, 290.3902588, -375.4851990, 370.8977661
1: -119.3920059, 282.0884705, -121.6965637, 288.1918945, -407.5838623, 403.7850342
2: -101.2751236, 310.7092590, -103.2365189, 317.3988342, -418.6739197, 413.9457703
3: -106.2089005, 403.8296509, -108.2730484, 412.3710938, -518.5800171, 512.1027222
4: -90.6926956, 367.2429504, -92.4498215, 374.8715820, -465.5642700, 459.6927795

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7683727, upper bound: 339.7935269
time: 1.22 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7683727, upper bound: 339.7935269
time: 1.14 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -85.5067825, 287.2420959, -85.0949707, 284.1704712, -369.6772461, 372.3370361
1: -120.1114273, 284.9082947, -119.3920059, 282.0884705, -402.1998596, 404.3002319
2: -101.8361511, 313.7402954, -101.2751236, 310.7092590, -412.5454102, 415.0153809
3: -106.8467102, 407.9730225, -106.2089005, 403.8296509, -510.6763611, 514.1819458
4: -91.2148666, 370.7578735, -90.6926956, 367.2429504, -458.4578247, 461.4505310

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7935269, upper bound: 339.7683727
time: 1.32 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7672412, upper bound: 339.7653524
time: 1.28 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -85.5067825, 287.2420959, -85.5067825, 287.2420959, -372.7488708, 372.7488708
1: -120.1114273, 284.9082947, -120.1114273, 284.9082947, -405.0196838, 405.0196838
2: -101.8361511, 313.7402954, -101.8361511, 313.7402954, -415.5764465, 415.5764465
3: -106.8467102, 407.9730225, -106.8467102, 407.9730225, -514.8197021, 514.8197021
4: -91.2148666, 370.7578735, -91.2148666, 370.7578735, -461.9727173, 461.9727173

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7935269, upper bound: 339.7703712
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7672412, upper bound: 339.7674305
time: 1.15 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.62 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.62
Output dim: 0, lower bound: -339.7683727, upper bound: 339.7935269
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.62
Output dim: 0, lower bound: -339.7683727, upper bound: 339.7935269
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.62
Output dim: 0, lower bound: -339.7935269, upper bound: 339.7683727
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.62
Output dim: 0, lower bound: -339.7672412, upper bound: 339.7653524
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.62
Output dim: 0, lower bound: -339.7935269, upper bound: 339.7703712
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.62
Output dim: 0, lower bound: -339.7672412, upper bound: 339.7674305

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -82.9748306, 276.9905396, -362.0854797, 367.1452637
1: -119.3920059, 282.0884705, -116.4064331, 275.0193787, -394.4113464, 398.4949036
2: -101.2751236, 310.7092590, -98.7795639, 302.9554138, -404.2305298, 409.4888000
3: -106.2089005, 403.8296509, -103.5627213, 393.5593262, -499.7682190, 507.3923645
4: -90.6926956, 367.2429504, -88.4737244, 357.8739319, -448.5666199, 455.7166443

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7678612, upper bound: 339.7934490
time: 1.18 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7679101, upper bound: 339.7935213
time: 1.40 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -83.4957504, 280.3916321, -365.4866028, 367.6661987
1: -119.3920059, 282.0884705, -117.3040619, 278.1459656, -397.5379639, 399.3925171
2: -101.2751236, 310.7092590, -99.4742203, 306.3189087, -407.5939941, 410.1834717
3: -106.2089005, 403.8296509, -104.3519974, 398.1978760, -504.4067688, 508.1816406
4: -90.6926956, 367.2429504, -89.1125412, 361.8850098, -452.5776978, 456.3554382

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7678612, upper bound: 339.7934490
time: 0.78 seconds

## Relational analysis of IS_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7679101, upper bound: 339.7935213
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -83.4957504, 280.3916321, -85.0949707, 284.1704712, -367.6661987, 365.4866028
1: -117.3040619, 278.1459656, -119.3920059, 282.0884705, -399.3925171, 397.5379639
2: -99.4742203, 306.3189087, -101.2751236, 310.7092590, -410.1834717, 407.5939941
3: -104.3519974, 398.1978760, -106.2089005, 403.8296509, -508.1816406, 504.4067688
4: -89.1125412, 361.8850098, -90.6926956, 367.2429504, -456.3554382, 452.5776978

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7934490, upper bound: 339.7678612
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7935213, upper bound: 339.7679101
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -83.4957504, 280.3916321, -85.5067825, 287.2420959, -370.7377625, 365.8984070
1: -117.3040619, 278.1459656, -120.1114273, 284.9082947, -402.2123413, 398.2573547
2: -99.4742203, 306.3189087, -101.8361511, 313.7402954, -413.2145081, 408.1550598
3: -104.3519974, 398.1978760, -106.8467102, 407.9730225, -512.3250122, 505.0445862
4: -89.1125412, 361.8850098, -91.2148666, 370.7578735, -459.8703613, 453.0998840

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7939722, upper bound: 339.7698656
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7938760, upper bound: 339.7698520
time: 0.84 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.83 seconds
IS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.83
Output dim: 0, lower bound: -339.7678612, upper bound: 339.7934490
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.83
Output dim: 0, lower bound: -339.7679101, upper bound: 339.7935213
IS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.83
Output dim: 0, lower bound: -339.7678612, upper bound: 339.7934490
IS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.83
Output dim: 0, lower bound: -339.7679101, upper bound: 339.7935213
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.83
Output dim: 0, lower bound: -339.7934490, upper bound: 339.7678612
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.83
Output dim: 0, lower bound: -339.7935213, upper bound: 339.7679101
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.83
Output dim: 0, lower bound: -339.7939722, upper bound: 339.7698656
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.83
Output dim: 0, lower bound: -339.7938760, upper bound: 339.7698520

## BFS IS instance: IS_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -79.7977753, 265.8059692, -350.9009094, 363.9682312
1: -119.3920059, 282.0884705, -111.9293900, 263.9555664, -383.3475647, 394.0178223
2: -101.2751236, 310.7092590, -95.0137329, 290.7597961, -392.0348511, 405.7229919
3: -106.2089005, 403.8296509, -99.5753555, 377.4585571, -483.6674500, 503.4049988
4: -90.6926956, 367.2429504, -85.1080856, 343.3711548, -434.0638428, 452.3510132

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7660885, upper bound: 339.7909255
time: 1.19 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7657662, upper bound: 339.7897239
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -84.9902344, 283.8057251, -86.9552078, 288.6056824, -373.5959167, 370.7609253
1: -119.2426605, 281.7272034, -122.0119247, 286.6141663, -405.8567505, 403.7391357
2: -101.1491013, 310.3151245, -103.5528488, 315.8373718, -416.9864807, 413.8679810
3: -106.0761871, 403.3096008, -108.4716873, 409.5214233, -515.5975952, 511.7812805
4: -90.5803604, 366.7749634, -92.6019135, 372.9360046, -463.5163574, 459.3768616

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7649837, upper bound: 339.7899582
time: 1.06 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7653707, upper bound: 339.7896866
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -80.4964371, 269.8757629, -354.9706726, 364.6668701
1: -119.3920059, 282.0884705, -113.0993881, 267.7301941, -387.1221313, 395.1878662
2: -101.2751236, 310.7092590, -95.9254608, 294.8403931, -396.1154480, 406.6346741
3: -106.2089005, 403.8296509, -100.6049500, 383.0876770, -489.2965698, 504.4346008
4: -90.6926956, 367.2429504, -85.9423904, 348.2578430, -438.9505310, 453.1853027

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B1_B1

### Relational analysis result of IS_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7643439, upper bound: 339.7895352
time: 1.03 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2

### Relational analysis result of IS_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7641483, upper bound: 339.7882861
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -84.9902344, 283.8057251, -87.6614304, 292.6433716, -377.6336060, 371.4671631
1: -119.2426605, 281.7272034, -123.1961212, 290.3698120, -409.6124268, 404.9233398
2: -101.1491013, 310.3151245, -104.4715958, 319.8901672, -421.0392761, 414.7867126
3: -106.0761871, 403.3096008, -109.5072021, 415.1761169, -521.2521973, 512.8167725
4: -90.5803604, 366.7749634, -93.4404907, 377.8494568, -468.4298096, 460.2154236

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7679101, upper bound: 339.7935213
time: 1.14 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7678612, upper bound: 339.7935213
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -80.4964371, 269.8757629, -85.0949707, 284.1704712, -364.6669006, 354.9706726
1: -113.0993881, 267.7301941, -119.3920059, 282.0884705, -395.1878662, 387.1221008
2: -95.9254608, 294.8403931, -101.2751236, 310.7092590, -406.6346741, 396.1154785
3: -100.6049500, 383.0876770, -106.2089005, 403.8296509, -504.4346008, 489.2965698
4: -85.9423904, 348.2578430, -90.6926956, 367.2429504, -453.1853027, 438.9505310

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7895352, upper bound: 339.7643439
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882861, upper bound: 339.7641483
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -87.6614304, 292.6433716, -84.9902344, 283.8057251, -371.4671631, 377.6336060
1: -123.1961212, 290.3698120, -119.2426605, 281.7272034, -404.9233398, 409.6124268
2: -104.4715958, 319.8901672, -101.1491013, 310.3151245, -414.7867126, 421.0392761
3: -109.5072021, 415.1761169, -106.0761871, 403.3096008, -512.8167725, 521.2521362
4: -93.4404907, 377.8494568, -90.5803604, 366.7749634, -460.2154236, 468.4298096

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7935213, upper bound: 339.7679101
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7935213, upper bound: 339.7679101
time: 1.45 seconds

## BFS IS instance: IS_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -80.4964371, 269.8757629, -85.5067825, 287.2420959, -367.7384644, 355.3825378
1: -113.0993881, 267.7301941, -120.1114273, 284.9082947, -398.0076904, 387.8415222
2: -95.9254608, 294.8403931, -101.8361511, 313.7402954, -409.6657104, 396.6765137
3: -100.6049500, 383.0876770, -106.8467102, 407.9730225, -508.5779724, 489.9343872
4: -85.9423904, 348.2578430, -91.2148666, 370.7578735, -456.7002258, 439.4727173

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7902741, upper bound: 339.7661432
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_A1_A2

### Relational analysis result of IS_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882861, upper bound: 339.7657708
time: 1.60 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -87.6614304, 292.6433716, -85.4039917, 286.8845825, -374.5460205, 378.0473633
1: -123.1961212, 290.3698120, -119.9648132, 284.5538635, -407.7500000, 410.3346252
2: -104.4715958, 319.8901672, -101.7121887, 313.3515320, -417.8231201, 421.6023254
3: -109.5072021, 415.1761169, -106.7164001, 407.4631042, -516.9703369, 521.8923340
4: -93.4404907, 377.8494568, -91.1043472, 370.2987061, -463.7391663, 468.9537964

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_A2_A1

### Relational analysis result of IS_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7901075, upper bound: 339.7661195
time: 1.25 seconds

## Relational analysis of IS_A2_B2_A1_A2_A2

### Relational analysis result of IS_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7887457, upper bound: 339.7659997
time: 0.91 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.31 seconds
IS_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.31
Output dim: 0, lower bound: -339.7660885, upper bound: 339.7909255
IS_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.31
Output dim: 0, lower bound: -339.7657662, upper bound: 339.7897239
IS_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.31
Output dim: 0, lower bound: -339.7649837, upper bound: 339.7899582
IS_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.31
Output dim: 0, lower bound: -339.7653707, upper bound: 339.7896866
IS_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.31
Output dim: 0, lower bound: -339.7643439, upper bound: 339.7895352
IS_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.31
Output dim: 0, lower bound: -339.7641483, upper bound: 339.7882861
IS_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.31
Output dim: 0, lower bound: -339.7679101, upper bound: 339.7935213
IS_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.31
Output dim: 0, lower bound: -339.7678612, upper bound: 339.7935213
IS_A2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.31
Output dim: 0, lower bound: -339.7895352, upper bound: 339.7643439
IS_A2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.31
Output dim: 0, lower bound: -339.7882861, upper bound: 339.7641483
IS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.31
Output dim: 0, lower bound: -339.7935213, upper bound: 339.7679101
IS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.31
Output dim: 0, lower bound: -339.7935213, upper bound: 339.7679101
IS_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.31
Output dim: 0, lower bound: -339.7902741, upper bound: 339.7661432
IS_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.31
Output dim: 0, lower bound: -339.7882861, upper bound: 339.7657708
IS_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.31
Output dim: 0, lower bound: -339.7901075, upper bound: 339.7661195
IS_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.31
Output dim: 0, lower bound: -339.7887457, upper bound: 339.7659997

## BFS IS instance: IS_A1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -84.7910843, 283.1136475, -74.0052567, 245.7193756, -330.5103760, 357.1188965
1: -118.9607544, 281.0505676, -103.2421646, 244.0806885, -363.0414429, 384.2927246
2: -100.9114532, 309.5650940, -87.7117767, 268.8742981, -369.7857666, 397.2768555
3: -105.8278961, 402.3255310, -91.9311142, 349.1697388, -454.9976196, 494.2566223
4: -90.3692093, 365.8819885, -78.6868591, 317.4967651, -407.8658752, 444.5688477

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7660885, upper bound: 339.7909255
time: 1.31 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7660885, upper bound: 339.7909255
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -73.8942566, 245.9733887, -331.0683594, 358.0647278
1: -119.3920059, 282.0884705, -103.4220047, 244.7231598, -364.1151123, 385.5104370
2: -101.2751236, 310.7092590, -87.7495346, 269.7869873, -371.0620728, 398.4588013
3: -106.2089005, 403.8296509, -92.1146774, 350.1335144, -456.3424072, 495.9443359
4: -90.6926956, 367.2429504, -78.7043076, 318.6690674, -409.3617249, 445.9472656

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7657662, upper bound: 339.7897239
time: 1.09 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7657662, upper bound: 339.7897239
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -84.6864700, 282.7495728, -82.5946960, 273.2151489, -357.9016113, 365.3442688
1: -118.8116455, 280.6898499, -115.5130005, 271.4657898, -390.2774353, 396.2028503
2: -100.7856064, 309.1690674, -98.0321426, 299.0640259, -399.8496399, 407.2012024
3: -105.6953735, 401.8063354, -102.7604294, 387.9916077, -493.6869812, 504.5667725
4: -90.2570114, 365.4147644, -87.6726685, 353.1832581, -443.4402771, 453.0874329

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7649837, upper bound: 339.7899582
time: 1.14 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7649837, upper bound: 339.7899582
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -84.9902344, 283.8057251, -81.0572510, 269.2322388, -354.2224731, 364.8629761
1: -119.2426605, 281.7272034, -113.5928268, 267.8582458, -387.1008911, 395.3200378
2: -101.1491013, 310.3151245, -96.3386841, 295.3662720, -396.5153809, 406.6537781
3: -106.0761871, 403.3096008, -101.1060562, 382.9111938, -488.9873657, 504.4156494
4: -90.5803604, 366.7749634, -86.2887878, 348.7875671, -439.3679199, 453.0637512

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7653707, upper bound: 339.7896866
time: 1.09 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7653707, upper bound: 339.7896866
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -84.7910843, 283.1136475, -74.5175018, 249.2518616, -334.0429382, 357.6311646
1: -118.9607544, 281.0505676, -104.1531906, 247.3020935, -366.2628479, 385.2037659
2: -100.9114532, 309.5650940, -88.3960266, 272.3433533, -373.2548218, 397.9611206
3: -105.8278961, 402.3255310, -92.7213516, 354.0706177, -459.8984985, 495.0468750
4: -90.3692093, 365.8819885, -79.3333664, 321.6880798, -412.0571899, 445.2153625

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7643439, upper bound: 339.7895352
time: 1.08 seconds

## Relational analysis of IS_A1_B1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7643439, upper bound: 339.7895352
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -74.6870956, 250.2789154, -335.3739014, 358.8575745
1: -119.3920059, 282.0884705, -104.7031937, 248.7222290, -368.1141357, 386.7916565
2: -101.2751236, 310.7092590, -88.7651215, 274.1075439, -375.3825684, 399.4743652
3: -106.2089005, 403.8296509, -93.2395935, 356.0269470, -462.2358398, 497.0692444
4: -90.6926956, 367.2429504, -79.6211777, 323.7671204, -414.4597473, 446.8641357

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7641483, upper bound: 339.7882861
time: 0.76 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7641483, upper bound: 339.7882861
time: 1.30 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -82.8679962, 276.6155396, -87.6614304, 292.6433716, -375.5113220, 364.2769775
1: -116.2538528, 274.6481628, -123.1961212, 290.3698120, -406.6236572, 397.8442993
2: -98.6508331, 302.5505066, -104.4715958, 319.8901672, -418.5410156, 407.0220947
3: -103.4270782, 393.0245667, -109.5072021, 415.1761169, -518.6031494, 502.5317688
4: -88.3587723, 357.3928528, -93.4404907, 377.8494568, -466.2082214, 450.8332825

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7644029, upper bound: 339.7895483
time: 1.06 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7643545, upper bound: 339.7887457
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -97.0391998, 328.6636658, -87.6614304, 292.6433716, -389.6825562, 416.3251038
1: -136.3375854, 325.6862183, -123.1961212, 290.3698120, -426.7073669, 448.8823242
2: -115.6902313, 358.5884399, -104.4715958, 319.8901672, -435.5803833, 463.0600281
3: -121.3332062, 465.8457031, -109.5072021, 415.1761169, -536.5092773, 575.3529053
4: -103.5656433, 422.9984436, -93.4404907, 377.8494568, -481.4150696, 516.4389648

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7595153, upper bound: 339.7889384
time: 0.94 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7551842, upper bound: 339.7870751
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -74.5175018, 249.2518616, -84.7910843, 283.1136475, -357.6311646, 334.0429382
1: -104.1531906, 247.3020935, -118.9607544, 281.0505676, -385.2037659, 366.2628479
2: -88.3960266, 272.3433533, -100.9114532, 309.5650940, -397.9611206, 373.2548218
3: -92.7213516, 354.0706177, -105.8278961, 402.3255310, -495.0468750, 459.8984985
4: -79.3333664, 321.6880798, -90.3692093, 365.8819885, -445.2153625, 412.0571899

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7895352, upper bound: 339.7643439
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7895352, upper bound: 339.7643439
time: 1.39 seconds

## BFS IS instance: IS_A2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -74.6870956, 250.2789154, -85.0949707, 284.1704712, -358.8575745, 335.3739014
1: -104.7031937, 248.7222290, -119.3920059, 282.0884705, -386.7916565, 368.1141357
2: -88.7651215, 274.1075439, -101.2751236, 310.7092590, -399.4743652, 375.3825684
3: -93.2395935, 356.0269470, -106.2089005, 403.8296509, -497.0692444, 462.2358398
4: -79.6211777, 323.7671204, -90.6926956, 367.2429504, -446.8641357, 414.4597473

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882861, upper bound: 339.7641483
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882861, upper bound: 339.7641483
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -87.6614304, 292.6433716, -82.8679962, 276.6155396, -364.2769775, 375.5113220
1: -123.1961212, 290.3698120, -116.2538528, 274.6481628, -397.8442993, 406.6236572
2: -104.4715958, 319.8901672, -98.6508331, 302.5505066, -407.0220947, 418.5410156
3: -109.5072021, 415.1761169, -103.4270782, 393.0245667, -502.5317688, 518.6031494
4: -93.4404907, 377.8494568, -88.3587723, 357.3928528, -450.8332825, 466.2082214

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7895483, upper bound: 339.7644029
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7887457, upper bound: 339.7643545
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -87.6614304, 292.6433716, -97.0391998, 328.6636658, -416.3251038, 389.6825562
1: -123.1961212, 290.3698120, -136.3375854, 325.6862183, -448.8823242, 426.7073669
2: -104.4715958, 319.8901672, -115.6902313, 358.5884399, -463.0600281, 435.5803833
3: -109.5072021, 415.1761169, -121.3332062, 465.8457031, -575.3529053, 536.5092773
4: -93.4404907, 377.8494568, -103.5656433, 422.9984436, -516.4389648, 481.4150696

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7889384, upper bound: 339.7595153
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7870751, upper bound: 339.7551842
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -74.5175018, 249.2518616, -85.1919174, 286.1514587, -360.6689453, 334.4437866
1: -104.1531906, 247.3020935, -119.6646576, 283.8367310, -387.9899292, 366.9667358
2: -88.3960266, 272.3433533, -101.4593887, 312.5623474, -400.9583740, 373.8027344
3: -92.7213516, 354.0706177, -106.4518967, 406.4251404, -499.1464844, 460.5224915
4: -79.3333664, 321.6880798, -90.8799896, 369.3542175, -448.6875916, 412.5680542

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7902741, upper bound: 339.7661432
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7902741, upper bound: 339.7661432
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -74.6870956, 250.2789154, -85.5067825, 287.2420959, -361.9291687, 335.7857056
1: -104.7031937, 248.7222290, -120.1114273, 284.9082947, -389.6114807, 368.8335266
2: -88.7651215, 274.1075439, -101.8361511, 313.7402954, -402.5054321, 375.9436340
3: -93.2395935, 356.0269470, -106.8467102, 407.9730225, -501.2126160, 462.8736572
4: -79.6211777, 323.7671204, -91.2148666, 370.7578735, -450.3790588, 414.9819336

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882861, upper bound: 339.7657708
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7882861, upper bound: 339.7657708
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -82.2580032, 273.6946106, -85.0893707, 285.7946777, -368.0526733, 358.7839966
1: -115.2041168, 271.6928101, -119.5183868, 283.4830017, -398.6871338, 391.2111816
2: -97.7027740, 299.2369080, -101.3357468, 312.1743164, -409.8770752, 400.5726624
3: -102.4676895, 388.6105042, -106.3218536, 405.9161987, -508.3838806, 494.9323425
4: -87.4080505, 353.4977722, -90.7697372, 368.8959961, -456.3040161, 444.2674255

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_A2_A1_B1

### Relational analysis result of IS_A2_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7901075, upper bound: 339.7661195
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A1_A2_A1_B2

### Relational analysis result of IS_A2_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7901075, upper bound: 339.7661195
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -81.8358765, 273.4550781, -85.4039917, 286.8845825, -368.7204285, 358.8590698
1: -114.8749313, 271.7871704, -119.9648132, 284.5538635, -399.4288025, 391.7519836
2: -97.3516312, 299.6034546, -101.7121887, 313.3515320, -410.7031555, 401.3156433
3: -102.2249146, 388.7427979, -106.7164001, 407.4631042, -509.6880188, 495.4591980
4: -87.1974792, 353.8367920, -91.1043472, 370.2987061, -457.4961853, 444.9411316

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_A2_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7887457, upper bound: 339.7659997
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_A2_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7887457, upper bound: 339.7659997
time: 0.90 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.50 seconds
IS_A1_B1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7660885, upper bound: 339.7909255
IS_A1_B1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7660885, upper bound: 339.7909255
IS_A1_B1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7657662, upper bound: 339.7897239
IS_A1_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7657662, upper bound: 339.7897239
IS_A1_B1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7649837, upper bound: 339.7899582
IS_A1_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7649837, upper bound: 339.7899582
IS_A1_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7653707, upper bound: 339.7896866
IS_A1_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7653707, upper bound: 339.7896866
IS_A1_B1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7643439, upper bound: 339.7895352
IS_A1_B1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7643439, upper bound: 339.7895352
IS_A1_B1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7641483, upper bound: 339.7882861
IS_A1_B1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7641483, upper bound: 339.7882861
IS_A1_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7644029, upper bound: 339.7895483
IS_A1_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7643545, upper bound: 339.7887457
IS_A1_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7595153, upper bound: 339.7889384
IS_A1_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7551842, upper bound: 339.7870751
IS_A2_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7895352, upper bound: 339.7643439
IS_A2_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7895352, upper bound: 339.7643439
IS_A2_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7882861, upper bound: 339.7641483
IS_A2_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7882861, upper bound: 339.7641483
IS_A2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7895483, upper bound: 339.7644029
IS_A2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7887457, upper bound: 339.7643545
IS_A2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7889384, upper bound: 339.7595153
IS_A2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7870751, upper bound: 339.7551842
IS_A2_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7902741, upper bound: 339.7661432
IS_A2_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7902741, upper bound: 339.7661432
IS_A2_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7882861, upper bound: 339.7657708
IS_A2_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7882861, upper bound: 339.7657708
IS_A2_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7901075, upper bound: 339.7661195
IS_A2_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7901075, upper bound: 339.7661195
IS_A2_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7887457, upper bound: 339.7659997
IS_A2_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.50
Output dim: 0, lower bound: -339.7887457, upper bound: 339.7659997

## BFS IS instance: IS_A1_B1_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -82.7080307, 276.0901794, -74.0052567, 245.7193756, -328.4273376, 350.0954285
1: -116.0296478, 274.1323242, -103.2421646, 244.0806885, -360.1103210, 377.3744812
2: -98.4615250, 301.9704285, -87.7117767, 268.8742981, -367.3358154, 389.6821899
3: -103.2297134, 392.2731018, -91.9311142, 349.1697388, -452.3994141, 484.2042236
4: -88.1900177, 356.7052917, -78.6868591, 317.4967651, -405.6867676, 435.3921509

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7609760, upper bound: 339.7902384
time: 1.02 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7549805, upper bound: 339.7790564
time: 2.01 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -96.8829498, 328.1154480, -74.0052567, 245.7193756, -342.6022949, 402.1206970
1: -136.1173706, 325.1502380, -103.2421646, 244.0806885, -380.1980286, 428.3923950
2: -115.5035706, 358.0007629, -87.7117767, 268.8742981, -384.3778076, 445.7125244
3: -121.1385117, 465.0728149, -91.9311142, 349.1697388, -470.3082275, 557.0038452
4: -103.3999252, 422.2928772, -78.6868591, 317.4967651, -420.8966675, 500.9797363

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7609760, upper bound: 339.7902384
time: 1.23 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7549805, upper bound: 339.7790564
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -82.9748306, 276.9905396, -73.8942566, 245.9733887, -328.9482117, 350.8847961
1: -116.4064331, 275.0193787, -103.4220047, 244.7231598, -361.1295776, 378.4413452
2: -98.7795639, 302.9554138, -87.7495346, 269.7869873, -368.5665283, 390.7049561
3: -103.5627213, 393.5593262, -92.1146774, 350.1335144, -453.6962280, 485.6740112
4: -88.4737244, 357.8739319, -78.7043076, 318.6690674, -407.1427612, 436.5782471

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7577383, upper bound: 339.7871385
time: 1.10 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7540047, upper bound: 339.7868121
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -97.1476517, 329.0505371, -73.8942566, 245.9733887, -343.1210327, 402.9447632
1: -136.4936829, 326.0688782, -103.4220047, 244.7231598, -381.2168579, 429.4908752
2: -115.8215332, 359.0078125, -87.7495346, 269.7869873, -385.6085205, 446.7573547
3: -121.4716415, 466.3991394, -92.1146774, 350.1335144, -471.6051636, 558.5137939
4: -103.6826248, 423.4954834, -78.7043076, 318.6690674, -422.3516235, 502.1997986

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7577383, upper bound: 339.7871385
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7540047, upper bound: 339.7868121
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -82.6015396, 275.7163086, -82.5946960, 273.2151489, -355.8166809, 358.3110046
1: -115.8775406, 273.7622986, -115.5130005, 271.4657898, -387.3433228, 389.2752991
2: -98.3332062, 301.5643921, -98.0321426, 299.0640259, -397.3972168, 399.5965271
3: -103.0944748, 391.7399597, -102.7604294, 387.9916077, -491.0860901, 494.5003967
4: -88.0754547, 356.2257690, -87.6726685, 353.1832581, -441.2587280, 443.8984375

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7598044, upper bound: 339.7892268
time: 0.96 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7562695, upper bound: 339.7862848
time: 1.25 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7643174, upper bound: 339.7898616
time: 1.23 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7648157, upper bound: 339.7898281
time: 1.37 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -96.7741013, 327.7266541, -82.5946960, 273.2151489, -369.9892578, 410.3213501
1: -135.9607239, 324.7656250, -115.5130005, 271.4657898, -407.4264221, 440.2786255
2: -115.3718033, 357.5792236, -98.0321426, 299.0640259, -414.4357300, 455.6113586
3: -120.9995193, 464.5163574, -102.7604294, 387.9916077, -508.9911194, 567.2767944
4: -103.2825623, 421.7930908, -87.6726685, 353.1832581, -456.4658203, 509.4657593

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7598044, upper bound: 339.7892268
time: 0.99 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7562695, upper bound: 339.7862848
time: 1.18 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7647254, upper bound: 339.7898291
time: 1.11 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7648157, upper bound: 339.7898281
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -82.8679962, 276.6155396, -81.0572510, 269.2322388, -352.1001892, 357.6727905
1: -116.2538528, 274.6481628, -113.5928268, 267.8582458, -384.1120911, 388.2409973
2: -98.6508331, 302.5505066, -96.3386841, 295.3662720, -394.0170898, 398.8891602
3: -103.4270782, 393.0245667, -101.1060562, 382.9111938, -486.3382568, 494.1306152
4: -88.3587723, 357.3928528, -86.2887878, 348.7875671, -437.1463318, 443.6816406

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7547147, upper bound: 339.7796715
time: 0.88 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7599697, upper bound: 339.7802282
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -97.0391998, 328.6636658, -81.0572510, 269.2322388, -366.2714233, 409.7209167
1: -136.3375854, 325.6862183, -113.5928268, 267.8582458, -404.1958313, 439.2790527
2: -115.6902313, 358.5884399, -96.3386841, 295.3662720, -411.0565186, 454.9271240
3: -121.3332062, 465.8457031, -101.1060562, 382.9111938, -504.2443848, 566.9517212
4: -103.5656433, 422.9984436, -86.2887878, 348.7875671, -452.3532104, 509.2872314

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7570906, upper bound: 339.7865445
time: 1.13 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7532900, upper bound: 339.7862405
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -82.7080307, 276.0901794, -74.5175018, 249.2518616, -331.9598999, 350.6076660
1: -116.0296478, 274.1323242, -104.1531906, 247.3020935, -363.3317261, 378.2855225
2: -98.4615250, 301.9704285, -88.3960266, 272.3433533, -370.8048706, 390.3664551
3: -103.2297134, 392.2731018, -92.7213516, 354.0706177, -457.3002930, 484.9944458
4: -88.1900177, 356.7052917, -79.3333664, 321.6880798, -409.8780823, 436.0386658

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7553564, upper bound: 339.7849943
time: 1.32 seconds

## Relational analysis of IS_A1_B1_B2_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7511298, upper bound: 339.7822403
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -96.8829498, 328.1154480, -74.5175018, 249.2518616, -346.1348267, 402.6329346
1: -136.1173706, 325.1502380, -104.1531906, 247.3020935, -383.4194641, 429.3034363
2: -115.5035706, 358.0007629, -88.3960266, 272.3433533, -387.8469238, 446.3967896
3: -121.1385117, 465.0728149, -92.7213516, 354.0706177, -475.2091064, 557.7941284
4: -103.3999252, 422.2928772, -79.3333664, 321.6880798, -425.0879822, 501.6262512

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7591239, upper bound: 339.7887336
time: 1.19 seconds

## Relational analysis of IS_A1_B1_B2_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7563883, upper bound: 339.7786469
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -82.9748306, 276.9905396, -74.6870956, 250.2789154, -333.2537231, 351.6776428
1: -116.4064331, 275.0193787, -104.7031937, 248.7222290, -365.1285400, 379.7225647
2: -98.7795639, 302.9554138, -88.7651215, 274.1075439, -372.8870239, 391.7205200
3: -103.5627213, 393.5593262, -93.2395935, 356.0269470, -459.5896606, 486.7989197
4: -88.4737244, 357.8739319, -79.6211777, 323.7671204, -412.2407532, 437.4951172

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7551124, upper bound: 339.7835398
time: 0.97 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7512277, upper bound: 339.7826940
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -97.1476517, 329.0505371, -74.6870956, 250.2789154, -347.4265747, 403.7376404
1: -136.4936829, 326.0688782, -104.7031937, 248.7222290, -385.2158203, 430.7720642
2: -115.8215332, 359.0078125, -88.7651215, 274.1075439, -389.9290771, 447.7729492
3: -121.4716415, 466.3991394, -93.2395935, 356.0269470, -477.4985962, 559.6387329
4: -103.6826248, 423.4954834, -79.6211777, 323.7671204, -427.4496460, 503.1166687

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7551124, upper bound: 339.7835398
time: 0.77 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7512277, upper bound: 339.7826940
time: 1.28 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -82.6015396, 275.7163086, -82.2580032, 273.6946106, -356.2961426, 357.9743042
1: -115.8775406, 273.7622986, -115.2041168, 271.6928101, -387.5703430, 388.9664307
2: -98.3332062, 301.5643921, -97.7027740, 299.2369080, -397.5701294, 399.2671509
3: -103.0944748, 391.7399597, -102.4676895, 388.6105042, -491.7049866, 494.2076416
4: -88.0754547, 356.2257690, -87.4080505, 353.4977722, -441.5732422, 443.6337585

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7925089, upper bound: 339.7937110
time: 1.06 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7925089, upper bound: 339.7937110
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -82.8679962, 276.6155396, -81.8358765, 273.4550781, -356.3230286, 358.4513550
1: -116.2538528, 274.6481628, -114.8749313, 271.7871704, -388.0410156, 389.5230713
2: -98.6508331, 302.5505066, -97.3516312, 299.6034546, -398.2542725, 399.9021301
3: -103.4270782, 393.0245667, -102.2249146, 388.7427979, -492.1698608, 495.2494812
4: -88.3587723, 357.3928528, -87.1974792, 353.8367920, -442.1955566, 444.5902710

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7923926, upper bound: 339.7920654
time: 1.01 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7923926, upper bound: 339.7924524
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -88.8458481, 303.5761108, -87.6614304, 292.6433716, -381.4892273, 391.2375488
1: -124.7456970, 300.4816895, -123.1961212, 290.3698120, -415.1155090, 423.6777954
2: -105.8136749, 330.8900146, -104.4715958, 319.8901672, -425.7038574, 435.3615723
3: -111.1215057, 430.4012756, -109.5072021, 415.1761169, -526.2975464, 539.9084473
4: -94.9147873, 390.6199341, -93.4404907, 377.8494568, -472.7642517, 484.0603943

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7553461, upper bound: 339.7847663
time: 0.95 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7552135, upper bound: 339.7835400
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -96.3473206, 326.2843628, -87.6614304, 292.6433716, -388.9906921, 413.9458008
1: -135.3759766, 323.3489380, -123.1961212, 290.3698120, -425.7457886, 446.5450439
2: -114.8838882, 356.0282288, -104.4715958, 319.8901672, -434.7740173, 460.4998169
3: -120.4812622, 462.4919128, -109.5072021, 415.1761169, -535.6571655, 571.9991455
4: -102.8488541, 419.9608154, -93.4404907, 377.8494568, -480.6983032, 513.4013062

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7465639, upper bound: 339.7853735
time: 1.34 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7448791, upper bound: 339.7780986
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -74.5175018, 249.2518616, -82.7080307, 276.0901794, -350.6076660, 331.9598999
1: -104.1531906, 247.3020935, -116.0296478, 274.1323242, -378.2855225, 363.3317261
2: -88.3960266, 272.3433533, -98.4615250, 301.9704285, -390.3664551, 370.8048706
3: -92.7213516, 354.0706177, -103.2297134, 392.2731018, -484.9944458, 457.3002930
4: -79.3333664, 321.6880798, -88.1900177, 356.7052917, -436.0386658, 409.8780823

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7849943, upper bound: 339.7553564
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A1_A1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7822403, upper bound: 339.7511298
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -74.5175018, 249.2518616, -96.8829498, 328.1154480, -402.6329346, 346.1348267
1: -104.1531906, 247.3020935, -136.1173706, 325.1502380, -429.3034363, 383.4194641
2: -88.3960266, 272.3433533, -115.5035706, 358.0007629, -446.3967896, 387.8469238
3: -92.7213516, 354.0706177, -121.1385117, 465.0728149, -557.7941284, 475.2091064
4: -79.3333664, 321.6880798, -103.3999252, 422.2928772, -501.6262207, 425.0879822

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_A1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7887336, upper bound: 339.7591239
time: 1.29 seconds

## Relational analysis of IS_A2_B1_A1_A1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7786469, upper bound: 339.7563883
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -74.6870956, 250.2789154, -82.9748306, 276.9905396, -351.6776428, 333.2537231
1: -104.7031937, 248.7222290, -116.4064331, 275.0193787, -379.7225647, 365.1285400
2: -88.7651215, 274.1075439, -98.7795639, 302.9554138, -391.7205200, 372.8870239
3: -93.2395935, 356.0269470, -103.5627213, 393.5593262, -486.7989197, 459.5896606
4: -79.6211777, 323.7671204, -88.4737244, 357.8739319, -437.4951172, 412.2407837

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7835398, upper bound: 339.7551124
time: 1.42 seconds

## Relational analysis of IS_A2_B1_A1_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7826940, upper bound: 339.7512277
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -74.6870956, 250.2789154, -97.1476517, 329.0505371, -403.7376404, 347.4265747
1: -104.7031937, 248.7222290, -136.4936829, 326.0688782, -430.7720642, 385.2158203
2: -88.7651215, 274.1075439, -115.8215332, 359.0078125, -447.7729492, 389.9290466
3: -93.2395935, 356.0269470, -121.4716415, 466.3991394, -559.6387329, 477.4985962
4: -79.6211777, 323.7671204, -103.6826248, 423.4954834, -503.1166687, 427.4496460

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7835398, upper bound: 339.7551124
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7826940, upper bound: 339.7512277
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -82.2580032, 273.6946106, -82.6015396, 275.7163086, -357.9743042, 356.2961426
1: -115.2041168, 271.6928101, -115.8775406, 273.7622986, -388.9664307, 387.5703430
2: -97.7027740, 299.2369080, -98.3332062, 301.5643921, -399.2671509, 397.5701294
3: -102.4676895, 388.6105042, -103.0944748, 391.7399597, -494.2076416, 491.7049866
4: -87.4080505, 353.4977722, -88.0754547, 356.2257690, -443.6337585, 441.5732422

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7937110, upper bound: 339.7925089
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7937110, upper bound: 339.7925089
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -81.8358765, 273.4550781, -82.8679962, 276.6155396, -358.4513550, 356.3230286
1: -114.8749313, 271.7871704, -116.2538528, 274.6481628, -389.5230713, 388.0410156
2: -97.3516312, 299.6034546, -98.6508331, 302.5505066, -399.9021301, 398.2542725
3: -102.2249146, 388.7427979, -103.4270782, 393.0245667, -495.2494812, 492.1698608
4: -87.1974792, 353.8367920, -88.3587723, 357.3928528, -444.5902710, 442.1955566

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7920654, upper bound: 339.7923926
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7920654, upper bound: 339.7923926
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -87.6614304, 292.6433716, -88.8458481, 303.5761108, -391.2375488, 381.4892273
1: -123.1961212, 290.3698120, -124.7456970, 300.4816895, -423.6777954, 415.1155090
2: -104.4715958, 319.8901672, -105.8136749, 330.8900146, -435.3615723, 425.7038574
3: -109.5072021, 415.1761169, -111.1215057, 430.4012756, -539.9084473, 526.2975464
4: -93.4404907, 377.8494568, -94.9147873, 390.6199341, -484.0603943, 472.7642517

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_A2_B2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7847663, upper bound: 339.7553461
time: 1.22 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7835400, upper bound: 339.7552135
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -87.6614304, 292.6433716, -96.3473206, 326.2843628, -413.9458008, 388.9906921
1: -123.1961212, 290.3698120, -135.3759766, 323.3489380, -446.5450439, 425.7457886
2: -104.4715958, 319.8901672, -114.8838882, 356.0282288, -460.4998169, 434.7740479
3: -109.5072021, 415.1761169, -120.4812622, 462.4919128, -571.9990845, 535.6572266
4: -93.4404907, 377.8494568, -102.8488541, 419.9608154, -513.4013062, 480.6983032

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_A2_B2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7853735, upper bound: 339.7465639
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7780986, upper bound: 339.7448791
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -74.5175018, 249.2518616, -83.2348557, 279.5156860, -354.0331421, 332.4867249
1: -104.1531906, 247.3020935, -116.9371567, 277.2814941, -381.4346924, 364.2392273
2: -88.3960266, 272.3433533, -99.1642075, 305.3659363, -393.7619629, 371.5075684
3: -92.7213516, 354.0706177, -104.0275650, 396.9531250, -489.6744690, 458.0981750
4: -79.3333664, 321.6880798, -88.8364105, 360.7524109, -440.0857849, 410.5244751

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7857253, upper bound: 339.7574921
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_A1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7860910, upper bound: 339.7640678
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_A1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7892088, upper bound: 339.7650273
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -74.5175018, 249.2518616, -96.0995865, 326.8689880, -401.3864746, 345.3514404
1: -104.1531906, 247.3020935, -135.1600800, 323.7302856, -427.8834839, 382.4621582
2: -88.3960266, 272.3433533, -114.6547012, 356.3714600, -444.7674866, 386.9980469
3: -92.7213516, 354.0706177, -120.2806549, 463.2255554, -555.9468994, 474.3512573
4: -79.3333664, 321.6880798, -102.6870041, 420.4258728, -499.7592468, 424.3750916

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_A1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7892010, upper bound: 339.7600260
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A1_A1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7783901, upper bound: 339.7570798
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -74.6870956, 250.2789154, -83.4957504, 280.3916321, -355.0787354, 333.7746277
1: -104.7031937, 248.7222290, -117.3040619, 278.1459656, -382.8491516, 366.0262146
2: -88.7651215, 274.1075439, -99.4742203, 306.3189087, -395.0840454, 373.5816956
3: -93.2395935, 356.0269470, -104.3519974, 398.1978760, -491.4374695, 460.3789368
4: -79.6211777, 323.7671204, -89.1125412, 361.8850098, -441.5061951, 412.8795471

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7800364, upper bound: 339.7590438
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A1_A1_A2_B1_B2

### Relational analysis result of IS_A2_B2_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7801206, upper bound: 339.7582927
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -74.6870956, 250.2789154, -96.3609467, 327.7865601, -402.4736633, 346.6398621
1: -104.7031937, 248.7222290, -135.5321503, 324.6314087, -429.3345947, 384.2542725
2: -88.7651215, 274.1075439, -114.9687119, 357.3590088, -446.1241455, 389.0762024
3: -93.2395935, 356.0269470, -120.6095428, 464.5240173, -557.7634888, 476.6364746
4: -79.6211777, 323.7671204, -102.9662323, 421.6016235, -501.2228088, 426.7332764

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_A1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7874278, upper bound: 339.7596506
time: 1.39 seconds

## Relational analysis of IS_A2_B2_A1_A1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7801206, upper bound: 339.7582927
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -82.2580032, 273.6946106, -83.1285477, 279.1425171, -361.4005127, 356.8231506
1: -115.2041168, 271.6928101, -116.7851257, 276.9122925, -392.1163940, 388.4779358
2: -97.7027740, 299.2369080, -99.0357590, 304.9609680, -402.6637268, 398.2726746
3: -102.4676895, 388.6105042, -103.8923874, 396.4209290, -498.8886108, 492.5028992
4: -87.4080505, 353.4977722, -88.7217178, 360.2736816, -447.6817017, 442.2194824

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7850496, upper bound: 339.7573972
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7892010, upper bound: 339.7600288
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7900506, upper bound: 339.7659593
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_A2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7900496, upper bound: 339.7660402
time: 1.29 seconds

## BFS IS instance: IS_A2_B2_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -82.2580032, 273.6946106, -95.9889755, 326.4730530, -408.7310486, 369.6835632
1: -115.2041168, 271.6928101, -135.0008545, 323.3397522, -438.5438843, 406.6936646
2: -97.7027740, 299.2369080, -114.5204391, 355.9429016, -453.6456604, 413.7573547
3: -102.4676895, 388.6105042, -120.1393814, 462.6598206, -565.1275024, 508.7498779
4: -87.4080505, 353.4977722, -102.5674973, 419.9180908, -507.3261108, 456.0652466

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_A2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7892010, upper bound: 339.7600288
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A1_A2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7850496, upper bound: 339.7573972
time: 1.12 seconds

## Relational analysis of IS_A2_B2_A1_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_A2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7900506, upper bound: 339.7659593
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_A2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7900496, upper bound: 339.7660402
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -81.8358765, 273.4550781, -83.3891907, 280.0175476, -361.8533630, 356.8442688
1: -114.8749313, 271.7871704, -117.1517334, 277.7760620, -392.6509705, 388.9388733
2: -97.3516312, 299.6034546, -99.3455353, 305.9130554, -403.2646790, 398.9489746
3: -102.2249146, 388.7427979, -104.2165527, 397.6647034, -499.8896179, 492.9593506
4: -87.1974792, 353.8367920, -88.9975967, 361.4052124, -448.6026611, 442.8343811

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7788382, upper bound: 339.7589792
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A1_A2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7790428, upper bound: 339.7582279
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -81.8358765, 273.4550781, -96.2504272, 327.3919373, -409.2277527, 369.7055054
1: -114.8749313, 271.7871704, -135.3730621, 324.2419128, -439.1168213, 407.1601868
2: -97.3516312, 299.6034546, -114.8345566, 356.9315186, -454.2831421, 414.4379578
3: -102.2249146, 388.7427979, -120.4683990, 463.9599915, -566.1848145, 509.2111816
4: -87.1974792, 353.8367920, -102.8467712, 421.0950623, -508.2925415, 456.6835632

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_A2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7835400, upper bound: 339.7571459
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_A2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7877394, upper bound: 339.7598120
time: 1.17 seconds

## Relational analysis of IS_A2_B2_A1_A2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7790428, upper bound: 339.7582279
time: 1.09 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 9.26 seconds
IS_A1_B1_B1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7609760, upper bound: 339.7902384
IS_A1_B1_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7549805, upper bound: 339.7790564
IS_A1_B1_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7609760, upper bound: 339.7902384
IS_A1_B1_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7549805, upper bound: 339.7790564
IS_A1_B1_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7577383, upper bound: 339.7871385
IS_A1_B1_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7540047, upper bound: 339.7868121
IS_A1_B1_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7577383, upper bound: 339.7871385
IS_A1_B1_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7540047, upper bound: 339.7868121
IS_A1_B1_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7643174, upper bound: 339.7898616
IS_A1_B1_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7648157, upper bound: 339.7898281
IS_A1_B1_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7647254, upper bound: 339.7898291
IS_A1_B1_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7648157, upper bound: 339.7898281
IS_A1_B1_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7547147, upper bound: 339.7796715
IS_A1_B1_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7599697, upper bound: 339.7802282
IS_A1_B1_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7570906, upper bound: 339.7865445
IS_A1_B1_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7532900, upper bound: 339.7862405
IS_A1_B1_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7553564, upper bound: 339.7849943
IS_A1_B1_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7511298, upper bound: 339.7822403
IS_A1_B1_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7591239, upper bound: 339.7887336
IS_A1_B1_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7563883, upper bound: 339.7786469
IS_A1_B1_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7551124, upper bound: 339.7835398
IS_A1_B1_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7512277, upper bound: 339.7826940
IS_A1_B1_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7551124, upper bound: 339.7835398
IS_A1_B1_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7512277, upper bound: 339.7826940
IS_A1_B1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7925089, upper bound: 339.7937110
IS_A1_B1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7925089, upper bound: 339.7937110
IS_A1_B1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7923926, upper bound: 339.7920654
IS_A1_B1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7923926, upper bound: 339.7924524
IS_A1_B1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7553461, upper bound: 339.7847663
IS_A1_B1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7552135, upper bound: 339.7835400
IS_A1_B1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7465639, upper bound: 339.7853735
IS_A1_B1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7448791, upper bound: 339.7780986
IS_A2_B1_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7849943, upper bound: 339.7553564
IS_A2_B1_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7822403, upper bound: 339.7511298
IS_A2_B1_A1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7887336, upper bound: 339.7591239
IS_A2_B1_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7786469, upper bound: 339.7563883
IS_A2_B1_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7835398, upper bound: 339.7551124
IS_A2_B1_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7826940, upper bound: 339.7512277
IS_A2_B1_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7835398, upper bound: 339.7551124
IS_A2_B1_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7826940, upper bound: 339.7512277
IS_A2_B1_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7937110, upper bound: 339.7925089
IS_A2_B1_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7937110, upper bound: 339.7925089
IS_A2_B1_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7920654, upper bound: 339.7923926
IS_A2_B1_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7920654, upper bound: 339.7923926
IS_A2_B1_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7847663, upper bound: 339.7553461
IS_A2_B1_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7835400, upper bound: 339.7552135
IS_A2_B1_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7853735, upper bound: 339.7465639
IS_A2_B1_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7780986, upper bound: 339.7448791
IS_A2_B2_A1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7860910, upper bound: 339.7640678
IS_A2_B2_A1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7892088, upper bound: 339.7650273
IS_A2_B2_A1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7892010, upper bound: 339.7600260
IS_A2_B2_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7783901, upper bound: 339.7570798
IS_A2_B2_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7800364, upper bound: 339.7590438
IS_A2_B2_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7801206, upper bound: 339.7582927
IS_A2_B2_A1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7874278, upper bound: 339.7596506
IS_A2_B2_A1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7801206, upper bound: 339.7582927
IS_A2_B2_A1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7900506, upper bound: 339.7659593
IS_A2_B2_A1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7900496, upper bound: 339.7660402
IS_A2_B2_A1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7900506, upper bound: 339.7659593
IS_A2_B2_A1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7900496, upper bound: 339.7660402
IS_A2_B2_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7788382, upper bound: 339.7589792
IS_A2_B2_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7790428, upper bound: 339.7582279
IS_A2_B2_A1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7877394, upper bound: 339.7598120
IS_A2_B2_A1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 9.26
Output dim: 0, lower bound: -339.7790428, upper bound: 339.7582279

## BFS IS instance: IS_A1_B1_B1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -82.6532211, 275.9209290, -66.6641235, 221.5711365, -304.2242737, 342.5850525
1: -115.9563370, 273.9618530, -93.1021500, 220.1848602, -336.1412048, 367.0639954
2: -98.3988037, 301.7829895, -79.0303879, 242.5976868, -340.9964905, 380.8133850
3: -103.1640167, 392.0309753, -82.9030151, 315.0477905, -418.2117920, 474.9339600
4: -88.1336288, 356.4845886, -70.8799667, 286.5773621, -374.7109375, 427.3645630

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_B1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7825134, upper bound: 339.7844727
time: 1.06 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7825134, upper bound: 339.7844727
time: 1.02 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 5.77 seconds
IS_A1_B1_B1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 5.77
Output dim: 0, lower bound: -339.7825134, upper bound: 339.7844727
IS_A1_B1_B1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 5.77
Output dim: 0, lower bound: -339.7825134, upper bound: 339.7844727
IS_A1_B1_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7549805, upper bound: 339.7790564
IS_A1_B1_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7609760, upper bound: 339.7902384
IS_A1_B1_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7549805, upper bound: 339.7790564
IS_A1_B1_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7577383, upper bound: 339.7871385
IS_A1_B1_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7540047, upper bound: 339.7868121
IS_A1_B1_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7577383, upper bound: 339.7871385
IS_A1_B1_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7540047, upper bound: 339.7868121
IS_A1_B1_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7643174, upper bound: 339.7898616
IS_A1_B1_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7648157, upper bound: 339.7898281
IS_A1_B1_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7647254, upper bound: 339.7898291
IS_A1_B1_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7648157, upper bound: 339.7898281
IS_A1_B1_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7547147, upper bound: 339.7796715
IS_A1_B1_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7599697, upper bound: 339.7802282
IS_A1_B1_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7570906, upper bound: 339.7865445
IS_A1_B1_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7532900, upper bound: 339.7862405
IS_A1_B1_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7553564, upper bound: 339.7849943
IS_A1_B1_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7511298, upper bound: 339.7822403
IS_A1_B1_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7591239, upper bound: 339.7887336
IS_A1_B1_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7563883, upper bound: 339.7786469
IS_A1_B1_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7551124, upper bound: 339.7835398
IS_A1_B1_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7512277, upper bound: 339.7826940
IS_A1_B1_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7551124, upper bound: 339.7835398
IS_A1_B1_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7512277, upper bound: 339.7826940
IS_A1_B1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7925089, upper bound: 339.7937110
IS_A1_B1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7925089, upper bound: 339.7937110
IS_A1_B1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7923926, upper bound: 339.7920654
IS_A1_B1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7923926, upper bound: 339.7924524
IS_A1_B1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7553461, upper bound: 339.7847663
IS_A1_B1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7552135, upper bound: 339.7835400
IS_A1_B1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7465639, upper bound: 339.7853735
IS_A1_B1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7448791, upper bound: 339.7780986
IS_A2_B1_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7849943, upper bound: 339.7553564
IS_A2_B1_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7822403, upper bound: 339.7511298
IS_A2_B1_A1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7887336, upper bound: 339.7591239
IS_A2_B1_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7786469, upper bound: 339.7563883
IS_A2_B1_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7835398, upper bound: 339.7551124
IS_A2_B1_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7826940, upper bound: 339.7512277
IS_A2_B1_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7835398, upper bound: 339.7551124
IS_A2_B1_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7826940, upper bound: 339.7512277
IS_A2_B1_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7937110, upper bound: 339.7925089
IS_A2_B1_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7937110, upper bound: 339.7925089
IS_A2_B1_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7920654, upper bound: 339.7923926
IS_A2_B1_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7920654, upper bound: 339.7923926
IS_A2_B1_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7847663, upper bound: 339.7553461
IS_A2_B1_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7835400, upper bound: 339.7552135
IS_A2_B1_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7853735, upper bound: 339.7465639
IS_A2_B1_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7780986, upper bound: 339.7448791
IS_A2_B2_A1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7860910, upper bound: 339.7640678
IS_A2_B2_A1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7892088, upper bound: 339.7650273
IS_A2_B2_A1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7892010, upper bound: 339.7600260
IS_A2_B2_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7783901, upper bound: 339.7570798
IS_A2_B2_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7800364, upper bound: 339.7590438
IS_A2_B2_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7801206, upper bound: 339.7582927
IS_A2_B2_A1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7874278, upper bound: 339.7596506
IS_A2_B2_A1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7801206, upper bound: 339.7582927
IS_A2_B2_A1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7900506, upper bound: 339.7659593
IS_A2_B2_A1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7900496, upper bound: 339.7660402
IS_A2_B2_A1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7900506, upper bound: 339.7659593
IS_A2_B2_A1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7900496, upper bound: 339.7660402
IS_A2_B2_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7788382, upper bound: 339.7589792
IS_A2_B2_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7790428, upper bound: 339.7582279
IS_A2_B2_A1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7877394, upper bound: 339.7598120
IS_A2_B2_A1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -339.7790428, upper bound: 339.7582279
Binary search (step 2): status=Status.UNKNOWN, low=0.2500000, high=0.3750000, mid=0.3750000, abs_max=385.80084228515625
rel_dist={0: [-339.8056838508241, 339.8056838508239]}

## Binary search (step 3) starts
Candidate diff: 0.3125000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8054904, upper bound: 339.8042672
time: 1.05 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
time: 1.56 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.82 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 2.82
Output dim: 0, lower bound: -339.8054904, upper bound: 339.8042672
IS_B2, status: Status.UNKNOWN, split count: 1, time: 2.82
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -88.6850433, 297.0671082, -85.0949707, 284.1704712, -372.8555298, 382.1620178
1: -124.4317169, 294.7806396, -119.3920059, 282.0884705, -406.5202026, 414.1725769
2: -105.5347290, 324.6318054, -101.2751236, 310.7092590, -416.2439575, 425.9068909
3: -110.7026138, 421.8989563, -106.2089005, 403.8296509, -514.5321655, 528.1078491
4: -94.4959106, 383.5209961, -90.6926956, 367.2429504, -461.7388306, 474.2136841

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
time: 1.18 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 0.98 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -88.6961975, 297.1046753, -85.5067825, 287.2420959, -375.9382324, 382.6114502
1: -124.4471970, 294.8176575, -120.1114273, 284.9082947, -409.3554993, 414.9290466
2: -105.5478058, 324.6724243, -101.8361511, 313.7402954, -419.2880859, 426.5085449
3: -110.7164154, 421.9519958, -106.8467102, 407.9730225, -518.6894531, 528.7987061
4: -94.5076294, 383.5692749, -91.2148666, 370.7578735, -465.2655029, 474.7841492

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 1.48 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
time: 0.97 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.97 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 4.97
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042184
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 4.97
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 4.97
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 4.97
Output dim: 0, lower bound: -339.8042184, upper bound: 339.8042183

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -85.0949707, 284.1704712, -369.2654419, 369.2654114
1: -119.3920059, 282.0884705, -119.3920059, 282.0884705, -401.4804077, 401.4804077
2: -101.2751236, 310.7092590, -101.2751236, 310.7092590, -411.9843445, 411.9843445
3: -106.2089005, 403.8296509, -106.2089005, 403.8296509, -510.0385437, 510.0385437
4: -90.6926956, 367.2429504, -90.6926956, 367.2429504, -457.9356384, 457.9356384

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921918, upper bound: 339.7680842
time: 0.91 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7651631
time: 1.07 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -85.5067825, 287.2420959, -85.0949707, 284.1704712, -369.6772461, 372.3370361
1: -120.1114273, 284.9082947, -119.3920059, 282.0884705, -402.1998596, 404.3002319
2: -101.8361511, 313.7402954, -101.2751236, 310.7092590, -412.5454102, 415.0153809
3: -106.8467102, 407.9730225, -106.2089005, 403.8296509, -510.6763611, 514.1819458
4: -91.2148666, 370.7578735, -90.6926956, 367.2429504, -458.4578247, 461.4505310

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921918, upper bound: 339.7680842
time: 1.08 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7653524
time: 0.79 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -85.5067825, 287.2420959, -372.3370361, 369.6772461
1: -119.3920059, 282.0884705, -120.1114273, 284.9082947, -404.3002319, 402.1998596
2: -101.2751236, 310.7092590, -101.8361511, 313.7402954, -415.0153809, 412.5454102
3: -106.2089005, 403.8296509, -106.8467102, 407.9730225, -514.1819458, 510.6763611
4: -90.6926956, 367.2429504, -91.2148666, 370.7578735, -461.4505615, 458.4578247

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7680842, upper bound: 339.7921918
time: 0.92 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7672412
time: 1.16 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -85.5067825, 287.2420959, -85.5067825, 287.2420959, -372.7488708, 372.7488708
1: -120.1114273, 284.9082947, -120.1114273, 284.9082947, -405.0196838, 405.0196838
2: -101.8361511, 313.7402954, -101.8361511, 313.7402954, -415.5764465, 415.5764465
3: -106.8467102, 407.9730225, -106.8467102, 407.9730225, -514.8197021, 514.8197021
4: -91.2148666, 370.7578735, -91.2148666, 370.7578735, -461.9727173, 461.9727173

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921918, upper bound: 339.7700979
time: 0.77 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7674305
time: 1.42 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.71 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 4.71
Output dim: 0, lower bound: -339.7921918, upper bound: 339.7680842
IS_B1_A1_A2, status: Status.VERIFIED, split count: 3, time: 4.71
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7651631
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.71
Output dim: 0, lower bound: -339.7921918, upper bound: 339.7680842
IS_B1_A2_A2, status: Status.VERIFIED, split count: 3, time: 4.71
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7653524
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.71
Output dim: 0, lower bound: -339.7680842, upper bound: 339.7921918
IS_B2_A1_B2, status: Status.VERIFIED, split count: 3, time: 4.71
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7672412
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.71
Output dim: 0, lower bound: -339.7921918, upper bound: 339.7700979
IS_B2_A2_A2, status: Status.VERIFIED, split count: 3, time: 4.71
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7674305

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -82.9748306, 276.9905396, -85.0949707, 284.1704712, -367.1452637, 362.0854797
1: -116.4064331, 275.0193787, -119.3920059, 282.0884705, -398.4949036, 394.4113464
2: -98.7795639, 302.9554138, -101.2751236, 310.7092590, -409.4888000, 404.2305298
3: -103.5627213, 393.5593262, -106.2089005, 403.8296509, -507.3923645, 499.7682190
4: -88.4737244, 357.8739319, -90.6926956, 367.2429504, -455.7166443, 448.5666199

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7651631
time: 0.99 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7651631
time: 1.03 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -83.4957504, 280.3916321, -85.0949707, 284.1704712, -367.6661987, 365.4866028
1: -117.3040619, 278.1459656, -119.3920059, 282.0884705, -399.3925171, 397.5379639
2: -99.4742203, 306.3189087, -101.2751236, 310.7092590, -410.1834717, 407.5939941
3: -104.3519974, 398.1978760, -106.2089005, 403.8296509, -508.1816406, 504.4067688
4: -89.1125412, 361.8850098, -90.6926956, 367.2429504, -456.3554382, 452.5776978

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_A1_A1

### Relational analysis result of IS_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7919832, upper bound: 339.7675476
time: 1.03 seconds

## Relational analysis of IS_B1_A2_A1_A2

### Relational analysis result of IS_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921918, upper bound: 339.7678016
time: 1.06 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -85.0949707, 284.1704712, -83.4957504, 280.3916321, -365.4866028, 367.6661987
1: -119.3920059, 282.0884705, -117.3040619, 278.1459656, -397.5379639, 399.3925171
2: -101.2751236, 310.7092590, -99.4742203, 306.3189087, -407.5939941, 410.1834717
3: -106.2089005, 403.8296509, -104.3519974, 398.1978760, -504.4067688, 508.1816406
4: -90.6926956, 367.2429504, -89.1125412, 361.8850098, -452.5776978, 456.3554382

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7675476, upper bound: 339.7919832
time: 1.14 seconds

## Relational analysis of IS_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7678016, upper bound: 339.7921918
time: 1.24 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -83.4957504, 280.3916321, -85.5067825, 287.2420959, -370.7377625, 365.8984070
1: -117.3040619, 278.1459656, -120.1114273, 284.9082947, -402.2123413, 398.2573547
2: -99.4742203, 306.3189087, -101.8361511, 313.7402954, -413.2145081, 408.1550598
3: -104.3519974, 398.1978760, -106.8467102, 407.9730225, -512.3250122, 505.0445862
4: -89.1125412, 361.8850098, -91.2148666, 370.7578735, -459.8703613, 453.0998840

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7674243, upper bound: 339.7674305
time: 1.19 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7674243, upper bound: 339.7674305
time: 1.19 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.41 seconds
IS_B1_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 5.41
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7651631
IS_B1_A1_A1_B2, status: Status.VERIFIED, split count: 4, time: 5.41
Output dim: 0, lower bound: -339.7651631, upper bound: 339.7651631
IS_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 0, lower bound: -339.7919832, upper bound: 339.7675476
IS_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 0, lower bound: -339.7921918, upper bound: 339.7678016
IS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 0, lower bound: -339.7675476, upper bound: 339.7919832
IS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 0, lower bound: -339.7678016, upper bound: 339.7921918
IS_B2_A2_A1_B1, status: Status.VERIFIED, split count: 4, time: 5.41
Output dim: 0, lower bound: -339.7674243, upper bound: 339.7674305
IS_B2_A2_A1_B2, status: Status.VERIFIED, split count: 4, time: 5.41
Output dim: 0, lower bound: -339.7674243, upper bound: 339.7674305

## BFS IS instance: IS_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -80.4964371, 269.8757629, -84.8136749, 283.1773682, -363.6737671, 354.6894531
1: -113.0993881, 267.7301941, -118.9967651, 281.1053162, -394.2047119, 386.7268066
2: -95.9254608, 294.8403931, -100.9428177, 309.6262817, -405.5516663, 395.7831421
3: -100.6049500, 383.0876770, -105.8567276, 402.3981018, -503.0030518, 488.9443970
4: -85.9423904, 348.2578430, -90.3955917, 365.9543152, -451.8966675, 438.6534424

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_A2_A1_A1_B1

### Relational analysis result of IS_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7919832, upper bound: 339.7675476
time: 1.22 seconds

## Relational analysis of IS_B1_A2_A1_A1_B2

### Relational analysis result of IS_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7919832, upper bound: 339.7675476
time: 1.07 seconds

## BFS IS instance: IS_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -87.6614304, 292.6433716, -84.5585175, 282.2896118, -369.9510498, 377.2019043
1: -123.1961212, 290.3698120, -118.6255951, 280.2271729, -403.4232788, 408.9953613
2: -104.4715958, 319.8901672, -100.6285248, 308.6784058, -413.1499939, 420.5186768
3: -109.5072021, 415.1761169, -105.5281372, 401.1476746, -510.6548767, 520.7041626
4: -93.4404907, 377.8494568, -90.1162109, 364.8308411, -458.2712708, 467.9656372

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_A2_A1_A2_B1

### Relational analysis result of IS_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921918, upper bound: 339.7678016
time: 1.24 seconds

## Relational analysis of IS_B1_A2_A1_A2_B2

### Relational analysis result of IS_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921918, upper bound: 339.7678016
time: 1.12 seconds

## BFS IS instance: IS_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -84.8136749, 283.1773682, -80.4964371, 269.8757629, -354.6894531, 363.6737366
1: -118.9967651, 281.1053162, -113.0993881, 267.7301941, -386.7268066, 394.2047119
2: -100.9428177, 309.6262817, -95.9254608, 294.8403931, -395.7832031, 405.5516663
3: -105.8567276, 402.3981018, -100.6049500, 383.0876770, -488.9443970, 503.0030518
4: -90.3955917, 365.9543152, -85.9423904, 348.2578430, -438.6534424, 451.8966675

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A1_B1_B1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7675476, upper bound: 339.7919832
time: 1.09 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7675476, upper bound: 339.7919832
time: 1.23 seconds

## BFS IS instance: IS_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -84.5585175, 282.2896118, -87.6614304, 292.6433716, -377.2019043, 369.9510498
1: -118.6255951, 280.2271729, -123.1961212, 290.3698120, -408.9953613, 403.4232788
2: -100.6285248, 308.6784058, -104.4715958, 319.8901672, -420.5186768, 413.1499939
3: -105.5281372, 401.1476746, -109.5072021, 415.1761169, -520.7041626, 510.6548767
4: -90.1162109, 364.8308411, -93.4404907, 377.8494568, -467.9656372, 458.2712708

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A1_B1_B2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7678016, upper bound: 339.7921918
time: 1.08 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2

### Relational analysis result of IS_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7678016, upper bound: 339.7921918
time: 1.21 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.37 seconds
IS_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 0, lower bound: -339.7919832, upper bound: 339.7675476
IS_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 0, lower bound: -339.7919832, upper bound: 339.7675476
IS_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 0, lower bound: -339.7921918, upper bound: 339.7678016
IS_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 0, lower bound: -339.7921918, upper bound: 339.7678016
IS_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 0, lower bound: -339.7675476, upper bound: 339.7919832
IS_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 0, lower bound: -339.7675476, upper bound: 339.7919832
IS_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 0, lower bound: -339.7678016, upper bound: 339.7921918
IS_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 0, lower bound: -339.7678016, upper bound: 339.7921918

## BFS IS instance: IS_B1_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -80.4964371, 269.8757629, -82.6955414, 276.0103149, -356.5067139, 352.5712280
1: -113.0993881, 267.7301941, -116.0149002, 274.0491028, -387.1484985, 383.7450562
2: -95.9254608, 294.8403931, -98.4506531, 301.8807068, -397.8061218, 393.2910156
3: -100.6049500, 383.0876770, -103.2139511, 392.1443481, -492.7492981, 486.3016357
4: -85.9423904, 348.2578430, -88.1798553, 356.5990601, -442.5414124, 436.4376831

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B1_A2_A1_A1_B1_A1

### Relational analysis result of IS_B1_A2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7867024, upper bound: 339.7636654
time: 1.10 seconds

## Relational analysis of IS_B1_A2_A1_A1_B1_A2

### Relational analysis result of IS_B1_A2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7875672, upper bound: 339.7639545
time: 1.07 seconds

## BFS IS instance: IS_B1_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -80.4964371, 269.8757629, -96.9011765, 328.1982727, -408.6946716, 366.7769470
1: -113.0993881, 267.7301941, -136.1496124, 325.2241211, -438.3235168, 403.8798218
2: -95.9254608, 294.8403931, -115.5318298, 358.0802307, -454.0056458, 410.3722229
3: -100.6049500, 383.0876770, -121.1648026, 465.1752625, -565.7802124, 504.2524719
4: -85.9423904, 348.2578430, -103.4232941, 422.3891602, -508.3315125, 451.6811218

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_A1_A1_B2_A1

### Relational analysis result of IS_B1_A2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7909375, upper bound: 339.7623059
time: 1.10 seconds

## Relational analysis of IS_B1_A2_A1_A1_B2_A2

### Relational analysis result of IS_B1_A2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7851507, upper bound: 339.7611524
time: 1.42 seconds

## BFS IS instance: IS_B1_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -87.6614304, 292.6433716, -82.4244995, 275.0574036, -362.7188416, 375.0678711
1: -123.1961212, 290.3698120, -115.6205521, 273.1060181, -396.3021240, 405.9903564
2: -104.4715958, 319.8901672, -98.1172180, 300.8687744, -405.3403625, 418.0073547
3: -109.5072021, 415.1761169, -102.8642654, 390.8040771, -500.3112793, 518.0402832
4: -93.4404907, 377.8494568, -87.8825836, 355.3960266, -448.8364868, 465.7320557

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B1_A2_A1_A2_B1_A1

### Relational analysis result of IS_B1_A2_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7670294, upper bound: 339.7522870
time: 1.36 seconds

## Relational analysis of IS_B1_A2_A1_A2_B1_A2

### Relational analysis result of IS_B1_A2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7878875, upper bound: 339.7642109
time: 1.09 seconds

## BFS IS instance: IS_B1_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -87.6614304, 292.6433716, -96.5650558, 326.9723206, -414.6337585, 389.2084045
1: -123.1961212, 290.3698120, -135.6550446, 324.0133057, -447.2094116, 426.0248413
2: -104.4715958, 319.8901672, -115.1160660, 356.7550049, -461.2265930, 435.0062256
3: -109.5072021, 415.1761169, -120.7278137, 463.4260559, -572.9332275, 535.9038696
4: -93.4404907, 377.8494568, -103.0541534, 420.8261108, -514.2666016, 480.9035950

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A2_A1_A2_B2_B1

### Relational analysis result of IS_B1_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7873657, upper bound: 339.7593451
time: 1.34 seconds

## Relational analysis of IS_B1_A2_A1_A2_B2_B2

### Relational analysis result of IS_B1_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7850918, upper bound: 339.7548177
time: 1.25 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -82.6955414, 276.0103149, -80.4964371, 269.8757629, -352.5712280, 356.5067139
1: -116.0149002, 274.0491028, -113.0993881, 267.7301941, -383.7450562, 387.1484985
2: -98.4506531, 301.8807068, -95.9254608, 294.8403931, -393.2910461, 397.8061218
3: -103.2139511, 392.1443481, -100.6049500, 383.0876770, -486.3016357, 492.7492981
4: -88.1798553, 356.5990601, -85.9423904, 348.2578430, -436.4376831, 442.5414124

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B2_A1_B1_B1_A1_B1

### Relational analysis result of IS_B2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7636654, upper bound: 339.7867024
time: 1.05 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_B2

### Relational analysis result of IS_B2_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7639545, upper bound: 339.7875672
time: 1.29 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -96.9011765, 328.1982727, -80.4964371, 269.8757629, -366.7769470, 408.6946716
1: -136.1496124, 325.2241211, -113.0993881, 267.7301941, -403.8798218, 438.3235168
2: -115.5318298, 358.0802307, -95.9254608, 294.8403931, -410.3722229, 454.0056458
3: -121.1648026, 465.1752625, -100.6049500, 383.0876770, -504.2524719, 565.7802124
4: -103.4232941, 422.3891602, -85.9423904, 348.2578430, -451.6811218, 508.3315125

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_B1_B1_A2_B1

### Relational analysis result of IS_B2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7623059, upper bound: 339.7909375
time: 1.14 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_B2

### Relational analysis result of IS_B2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7611524, upper bound: 339.7851507
time: 1.14 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -82.4244995, 275.0574036, -87.6614304, 292.6433716, -375.0678711, 362.7188416
1: -115.6205521, 273.1060181, -123.1961212, 290.3698120, -405.9903564, 396.3021240
2: -98.1172180, 300.8687744, -104.4715958, 319.8901672, -418.0073547, 405.3403625
3: -102.8642654, 390.8040771, -109.5072021, 415.1761169, -518.0402222, 500.3112793
4: -87.8825836, 355.3960266, -93.4404907, 377.8494568, -465.7320557, 448.8364868

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B2_A1_B1_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7522870, upper bound: 339.7670294
time: 1.69 seconds

## Relational analysis of IS_B2_A1_B1_B2_A1_B2

### Relational analysis result of IS_B2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7642109, upper bound: 339.7878875
time: 1.00 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -96.5650558, 326.9723206, -87.6614304, 292.6433716, -389.2084045, 414.6337585
1: -135.6550446, 324.0133057, -123.1961212, 290.3698120, -426.0248413, 447.2094116
2: -115.1160660, 356.7550049, -104.4715958, 319.8901672, -435.0062256, 461.2265930
3: -120.7278137, 463.4260559, -109.5072021, 415.1761169, -535.9038696, 572.9332275
4: -103.0541534, 420.8261108, -93.4404907, 377.8494568, -480.9035950, 514.2666016

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_B1_B2_A2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7593451, upper bound: 339.7873657
time: 0.74 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2_A2

### Relational analysis result of IS_B2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7548177, upper bound: 339.7850918
time: 1.54 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.20 seconds
IS_B1_A2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 0, lower bound: -339.7867024, upper bound: 339.7636654
IS_B1_A2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 0, lower bound: -339.7875672, upper bound: 339.7639545
IS_B1_A2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 0, lower bound: -339.7909375, upper bound: 339.7623059
IS_B1_A2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 0, lower bound: -339.7851507, upper bound: 339.7611524
IS_B1_A2_A1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 0, lower bound: -339.7670294, upper bound: 339.7522870
IS_B1_A2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 0, lower bound: -339.7878875, upper bound: 339.7642109
IS_B1_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 0, lower bound: -339.7873657, upper bound: 339.7593451
IS_B1_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 0, lower bound: -339.7850918, upper bound: 339.7548177
IS_B2_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 0, lower bound: -339.7636654, upper bound: 339.7867024
IS_B2_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 0, lower bound: -339.7639545, upper bound: 339.7875672
IS_B2_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 0, lower bound: -339.7623059, upper bound: 339.7909375
IS_B2_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 0, lower bound: -339.7611524, upper bound: 339.7851507
IS_B2_A1_B1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 0, lower bound: -339.7522870, upper bound: 339.7670294
IS_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 0, lower bound: -339.7642109, upper bound: 339.7878875
IS_B2_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 0, lower bound: -339.7593451, upper bound: 339.7873657
IS_B2_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 0, lower bound: -339.7548177, upper bound: 339.7850918

## BFS IS instance: IS_B1_A2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -74.5175018, 249.2518616, -82.2983170, 274.6703491, -349.1878357, 331.5501709
1: -104.1531906, 247.3020935, -115.4537888, 272.7283936, -376.8815918, 362.7558899
2: -88.3960266, 272.3433533, -97.9769363, 300.4215698, -388.8175964, 370.3202820
3: -92.7213516, 354.0706177, -102.7181015, 390.2294617, -482.9507751, 456.7887268
4: -79.3333664, 321.6880798, -87.7573242, 354.8599548, -434.1933289, 409.4453125

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_A1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7958247, upper bound: 339.7929008
time: 0.88 seconds

## Relational analysis of IS_B1_A2_A1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7958247, upper bound: 339.7929008
time: 0.97 seconds

## BFS IS instance: IS_B1_A2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -74.6870956, 250.2789154, -82.2494507, 274.5165100, -349.2036133, 332.5283508
1: -104.7031937, 248.7222290, -115.3730850, 272.5921631, -377.2953491, 364.0952454
2: -88.7651215, 274.1075439, -97.9007568, 300.3065186, -389.0716553, 372.0082397
3: -93.2395935, 356.0269470, -102.6527328, 390.0871277, -483.3267212, 458.6796570
4: -79.6211777, 323.7671204, -87.6986313, 354.7337036, -434.3548889, 411.4656982

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A2_A1_A1_B1_A2_B1

### Relational analysis result of IS_B1_A2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7916058, upper bound: 339.7921709
time: 1.11 seconds

## Relational analysis of IS_B1_A2_A1_A1_B1_A2_B2

### Relational analysis result of IS_B1_A2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7916058, upper bound: 339.7921709
time: 0.94 seconds

## BFS IS instance: IS_B1_A2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -72.3416290, 243.0059357, -96.6944962, 327.5869141, -399.9285278, 339.7004089
1: -101.6895752, 241.0612793, -135.8842926, 324.6145935, -426.3041382, 376.9455261
2: -86.2042160, 265.4900513, -115.3048782, 357.4085999, -443.6128235, 380.7949219
3: -90.4386978, 345.0323486, -120.9274597, 464.3043518, -554.7430420, 465.9597473
4: -77.2775726, 313.6072998, -103.2208023, 421.5929565, -498.8705444, 416.8280640

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_A1_A1_B2_A1_A1

### Relational analysis result of IS_B1_A2_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7893256, upper bound: 339.7618467
time: 0.97 seconds

## Relational analysis of IS_B1_A2_A1_A1_B2_A1_A2

### Relational analysis result of IS_B1_A2_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7908610, upper bound: 339.7619682
time: 1.04 seconds

## BFS IS instance: IS_B1_A2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -76.1799774, 256.2743530, -96.4398956, 326.7180176, -402.8979797, 352.7141724
1: -106.8986206, 254.1598816, -135.4880219, 323.7489319, -430.6474915, 389.6478882
2: -90.6461029, 279.8891907, -114.9678345, 356.4515991, -447.0977173, 394.8569336
3: -95.1265259, 363.9190063, -120.5776291, 463.0901794, -558.2166748, 484.4966431
4: -81.2976074, 330.6740417, -102.9204407, 420.4821472, -501.7797546, 433.5944824

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A2_A1_A1_B2_A2_B1

### Relational analysis result of IS_B1_A2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7805460, upper bound: 339.7478847
time: 2.01 seconds

## Relational analysis of IS_B1_A2_A1_A1_B2_A2_B2

### Relational analysis result of IS_B1_A2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7804902, upper bound: 339.7450671
time: 1.13 seconds

## BFS IS instance: IS_B1_A2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -81.8358765, 273.4550781, -81.9807968, 273.5749207, -355.4107361, 355.4358826
1: -114.8749313, 271.7871704, -114.9824219, 271.6639099, -386.5388489, 386.7695923
2: -97.3516312, 299.6034546, -97.5703659, 299.3084412, -396.6600647, 397.1738281
3: -102.2249146, 388.7427979, -102.3063965, 388.7633667, -490.9882812, 491.0491943
4: -87.1974792, 353.8367920, -87.4042511, 353.5455322, -440.7429810, 441.2410278

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A2_A1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7920654, upper bound: 339.7923926
time: 1.04 seconds

## Relational analysis of IS_B1_A2_A1_A2_B1_A2_B2

### Relational analysis result of IS_B1_A2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7920654, upper bound: 339.7923926
time: 1.20 seconds

## BFS IS instance: IS_B1_A2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -87.6614304, 292.6433716, -88.4092865, 302.0166321, -389.6780701, 381.0526123
1: -123.1961212, 290.3698120, -124.1170578, 298.9455261, -422.1416626, 414.4868774
2: -104.4715958, 319.8901672, -105.2842484, 329.2030029, -433.6745911, 425.1744080
3: -109.5072021, 415.1761169, -110.5639496, 428.1783752, -537.6855469, 525.7399292
4: -93.4404907, 377.8494568, -94.4438477, 388.6231079, -482.0635376, 472.2933044

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B1_A2_A1_A2_B2_B1_A1

### Relational analysis result of IS_B1_A2_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7590247, upper bound: 339.7404968
time: 1.25 seconds

## Relational analysis of IS_B1_A2_A1_A2_B2_B1_A2

### Relational analysis result of IS_B1_A2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7828511, upper bound: 339.7551176
time: 1.21 seconds

## BFS IS instance: IS_B1_A2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -87.6614304, 292.6433716, -95.8653870, 324.5655212, -412.2269592, 388.5087585
1: -123.1961212, 290.3698120, -134.6848450, 321.6502075, -444.8463135, 425.0546265
2: -104.4715958, 319.8901672, -114.3025360, 354.1665344, -458.6381226, 434.1926575
3: -109.5072021, 415.1761169, -119.8681030, 460.0350037, -569.5421143, 535.0441284
4: -93.4404907, 377.8494568, -102.3310471, 417.7551270, -511.1955872, 480.1804810

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_A1_A2_B2_B2_A1

### Relational analysis result of IS_B1_A2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7830085, upper bound: 339.7460819
time: 1.07 seconds

## Relational analysis of IS_B1_A2_A1_A2_B2_B2_A2

### Relational analysis result of IS_B1_A2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7780986, upper bound: 339.7448691
time: 0.96 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -82.2983170, 274.6703491, -74.5175018, 249.2518616, -331.5501709, 349.1878357
1: -115.4537888, 272.7283936, -104.1531906, 247.3020935, -362.7558899, 376.8815918
2: -97.9769363, 300.4215698, -88.3960266, 272.3433533, -370.3202820, 388.8175964
3: -102.7181015, 390.2294617, -92.7213516, 354.0706177, -456.7887268, 482.9507751
4: -87.7573242, 354.8599548, -79.3333664, 321.6880798, -409.4453125, 434.1933289

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_B1_B1_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7929008, upper bound: 339.7958247
time: 1.27 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7929008, upper bound: 339.7958247
time: 1.12 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -82.2494507, 274.5165100, -74.6870956, 250.2789154, -332.5283508, 349.2036133
1: -115.3730850, 272.5921631, -104.7031937, 248.7222290, -364.0952759, 377.2953491
2: -97.9007568, 300.3065186, -88.7651215, 274.1075439, -372.0082397, 389.0716553
3: -102.6527328, 390.0871277, -93.2395935, 356.0269470, -458.6796570, 483.3267212
4: -87.6986313, 354.7337036, -79.6211777, 323.7671204, -411.4656677, 434.3548889

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_A1_B1_B1_A1_B2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921709, upper bound: 339.7916058
time: 1.15 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_B2_A2

### Relational analysis result of IS_B2_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7921709, upper bound: 339.7919928
time: 1.17 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -96.6944962, 327.5869141, -72.3416290, 243.0059357, -339.7004089, 399.9285278
1: -135.8842926, 324.6145935, -101.6895752, 241.0612793, -376.9455261, 426.3041382
2: -115.3048782, 357.4085999, -86.2042160, 265.4900513, -380.7949219, 443.6128235
3: -120.9274597, 464.3043518, -90.4386978, 345.0323486, -465.9597778, 554.7430420
4: -103.2208023, 421.5929565, -77.2775726, 313.6072998, -416.8280640, 498.8705444

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_B1_B1_A2_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7618467, upper bound: 339.7893256
time: 1.39 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_B1_B2

### Relational analysis result of IS_B2_A1_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7619682, upper bound: 339.7908610
time: 1.06 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -96.4398956, 326.7180176, -76.1799774, 256.2743530, -352.7141724, 402.8980103
1: -135.4880219, 323.7489319, -106.8986206, 254.1598816, -389.6478882, 430.6474915
2: -114.9678345, 356.4515991, -90.6461029, 279.8891907, -394.8569336, 447.0977173
3: -120.5776291, 463.0901794, -95.1265259, 363.9190063, -484.4966431, 558.2166748
4: -102.9204407, 420.4821472, -81.2976074, 330.6740417, -433.5944824, 501.7797546

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_B1_B1_A2_B2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7478847, upper bound: 339.7805460
time: 0.88 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_B2_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7450671, upper bound: 339.7804902
time: 1.08 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -81.9807968, 273.5749207, -81.8358765, 273.4550781, -355.4358826, 355.4107361
1: -114.9824219, 271.6639099, -114.8749313, 271.7871704, -386.7695923, 386.5388489
2: -97.5703659, 299.3084412, -97.3516312, 299.6034546, -397.1738281, 396.6600647
3: -102.3063965, 388.7633667, -102.2249146, 388.7427979, -491.0491943, 490.9882812
4: -87.4042511, 353.5455322, -87.1974792, 353.8367920, -441.2410278, 440.7430115

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_A1_B1_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7923926, upper bound: 339.7920654
time: 0.94 seconds

## Relational analysis of IS_B2_A1_B1_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7923926, upper bound: 339.7924524
time: 0.97 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -88.4092865, 302.0166321, -87.6614304, 292.6433716, -381.0526123, 389.6780701
1: -124.1170578, 298.9455261, -123.1961212, 290.3698120, -414.4868774, 422.1416626
2: -105.2842484, 329.2030029, -104.4715958, 319.8901672, -425.1744080, 433.6745911
3: -110.5639496, 428.1783752, -109.5072021, 415.1761169, -525.7399292, 537.6855469
4: -94.4438477, 388.6231079, -93.4404907, 377.8494568, -472.2933044, 482.0635376

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B2_A1_B1_B2_A2_A1_B1

### Relational analysis result of IS_B2_A1_B1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -339.7404968, upper bound: 339.7590247
time: 1.12 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2_A1_B2

### Relational analysis result of IS_B2_A1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7551176, upper bound: 339.7828511
time: 1.01 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -95.8653870, 324.5655212, -87.6614304, 292.6433716, -388.5087585, 412.2269592
1: -134.6848450, 321.6502075, -123.1961212, 290.3698120, -425.0546265, 444.8463135
2: -114.3025360, 354.1665344, -104.4715958, 319.8901672, -434.1926575, 458.6381226
3: -119.8681030, 460.0350037, -109.5072021, 415.1761169, -535.0441284, 569.5421143
4: -102.3310471, 417.7551270, -93.4404907, 377.8494568, -480.1804810, 511.1955872

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_B1_B2_A2_A2_B1

### Relational analysis result of IS_B2_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7460819, upper bound: 339.7830085
time: 1.14 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2_A2_B2

### Relational analysis result of IS_B2_A1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -339.7448691, upper bound: 339.7780986
time: 1.32 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.64 seconds
IS_B1_A2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7958247, upper bound: 339.7929008
IS_B1_A2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7958247, upper bound: 339.7929008
IS_B1_A2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7916058, upper bound: 339.7921709
IS_B1_A2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7916058, upper bound: 339.7921709
IS_B1_A2_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7893256, upper bound: 339.7618467
IS_B1_A2_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7908610, upper bound: 339.7619682
IS_B1_A2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7805460, upper bound: 339.7478847
IS_B1_A2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7804902, upper bound: 339.7450671
IS_B1_A2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7920654, upper bound: 339.7923926
IS_B1_A2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7920654, upper bound: 339.7923926
IS_B1_A2_A1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7590247, upper bound: 339.7404968
IS_B1_A2_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7828511, upper bound: 339.7551176
IS_B1_A2_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7830085, upper bound: 339.7460819
IS_B1_A2_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7780986, upper bound: 339.7448691
IS_B2_A1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7929008, upper bound: 339.7958247
IS_B2_A1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7929008, upper bound: 339.7958247
IS_B2_A1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7921709, upper bound: 339.7916058
IS_B2_A1_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7921709, upper bound: 339.7919928
IS_B2_A1_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7618467, upper bound: 339.7893256
IS_B2_A1_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7619682, upper bound: 339.7908610
IS_B2_A1_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7478847, upper bound: 339.7805460
IS_B2_A1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7450671, upper bound: 339.7804902
IS_B2_A1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7923926, upper bound: 339.7920654
IS_B2_A1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7923926, upper bound: 339.7924524
IS_B2_A1_B1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7404968, upper bound: 339.7590247
IS_B2_A1_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7551176, upper bound: 339.7828511
IS_B2_A1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7460819, upper bound: 339.7830085
IS_B2_A1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.64
Output dim: 0, lower bound: -339.7448691, upper bound: 339.7780986

## BFS IS instance: IS_B1_A2_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -74.5175018, 249.2518616, -79.3901749, 264.4360352, -338.9535522, 328.6420288
1: -104.1531906, 247.3020935, -111.3564911, 262.5981445, -366.7513428, 358.6585693
2: -88.3960266, 272.3433533, -94.5284119, 289.2695007, -377.6655273, 366.8717651
3: -92.7213516, 354.0706177, -99.0677109, 375.5018616, -468.2232056, 453.1382751
4: -79.3333664, 321.6880798, -84.6750259, 341.5939331, -420.9273071, 406.3630981

Time for backsubstitution: 2.40 seconds
Binary search (step 3): status=Status.UNKNOWN, low=0.2500000, high=0.3125000, mid=0.3125000, abs_max=385.80084228515625
rel_dist={0: [-339.80564424607314, 339.80564424607314]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.25
execution time: 1105.25 seconds
