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
execution time: IAR + LP analysis = 2.07 + 2.20 = 4.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -554.9912953, upper bound: 554.9912953


# Binary Search by BASE starts (time budget: 1195.73 seconds, max iter: 100)

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
Binary search time: 88.69 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1107.04 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9842844, upper bound: 554.9869321
time: 1.07 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9854117, upper bound: 554.9854117
time: 0.91 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.15 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.15
Output dim: 0, lower bound: -554.9842844, upper bound: 554.9869321
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.15
Output dim: 0, lower bound: -554.9854117, upper bound: 554.9854117

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -144.4209442, 458.6982422, -148.1843414, 471.4650574, -615.8859863, 606.8823853
1: -202.9677582, 461.6006775, -208.4494476, 474.2722168, -677.2398071, 670.0501099
2: -171.4153137, 510.5334778, -176.0516663, 524.4273071, -695.8425293, 686.5851440
3: -180.9070740, 655.8257446, -185.7463837, 673.9080200, -854.8150024, 841.5721436
4: -154.0161438, 599.6799927, -158.1413727, 615.8510742, -769.8671875, 757.8212891

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9784586, upper bound: 554.9792687
time: 1.06 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9712994, upper bound: 554.9737705
time: 1.02 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -147.6497955, 467.7235718, -146.6156921, 466.3266296, -613.9764404, 614.3391724
1: -206.9226227, 471.0274048, -206.1553040, 469.1430969, -676.0656128, 677.1827393
2: -174.7036133, 521.2171631, -174.1037903, 518.8017578, -693.5053711, 695.3209229
3: -184.5714264, 669.6788330, -183.7223969, 666.6743774, -851.2457886, 853.4012451
4: -157.0737762, 613.0461426, -156.4145508, 609.2911377, -766.3648682, 769.4605103

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9795859, upper bound: 554.9779163
time: 1.24 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723812, upper bound: 554.9723813
time: 1.14 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.22 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 4.22
Output dim: 0, lower bound: -554.9784586, upper bound: 554.9792687
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 4.22
Output dim: 0, lower bound: -554.9712994, upper bound: 554.9737705
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 4.22
Output dim: 0, lower bound: -554.9795859, upper bound: 554.9779163
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 4.22
Output dim: 0, lower bound: -554.9723812, upper bound: 554.9723813

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -131.8264923, 415.5336609, -148.0532990, 471.0341492, -602.8606567, 563.5869751
1: -184.7457886, 418.9423218, -208.2655029, 473.8415222, -658.5872803, 627.2077026
2: -156.0319977, 463.5839844, -175.8965302, 523.9502563, -679.9820557, 639.4805298
3: -164.7066956, 594.9381104, -185.5812531, 673.2926025, -837.9991455, 780.5192871
4: -140.2548218, 544.5361328, -158.0018616, 615.2876587, -755.5423584, 702.5379639

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9712994, upper bound: 554.9737705
time: 0.98 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9712994, upper bound: 554.9737705
time: 0.96 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -138.0580139, 439.1935120, -148.1843414, 471.4650574, -609.5230713, 587.3778687
1: -194.1572723, 441.8686218, -208.4494476, 474.2722168, -668.4293213, 650.3180542
2: -163.9659271, 488.6260071, -176.0516663, 524.4273071, -688.3932495, 664.6776733
3: -173.0027618, 627.9880371, -185.7463837, 673.9080200, -846.9106445, 813.7344360
4: -147.3025970, 573.9549561, -158.1413727, 615.8510742, -763.1536865, 732.0962524

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9712994, upper bound: 554.9737705
time: 1.06 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9712994, upper bound: 554.9737705
time: 1.24 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -134.9726562, 424.7419739, -146.4843292, 465.8937683, -600.8664551, 571.2263184
1: -188.6135559, 428.4528198, -205.9705200, 468.7106323, -657.3241577, 634.4233398
2: -159.2701263, 474.2145996, -173.9482574, 518.3224487, -677.5925293, 648.1628418
3: -168.2837219, 608.8666992, -183.5567780, 666.0563354, -834.3400879, 792.4234619
4: -143.2449951, 557.9903564, -156.2746887, 608.7249146, -751.9698486, 714.2650146

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9788157, upper bound: 554.9759393
time: 0.99 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9786611, upper bound: 554.9758636
time: 0.94 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -141.6680908, 449.4027405, -146.6156921, 466.3266296, -607.9947510, 596.0184326
1: -198.6617737, 452.4513550, -206.1553040, 469.1430969, -667.8048706, 658.6066284
2: -167.7134552, 500.6373596, -174.1037903, 518.8017578, -686.5151978, 674.7410889
3: -177.1431580, 643.5125732, -183.7223969, 666.6743774, -843.8174438, 827.2349854
4: -150.7537537, 588.8465576, -156.4145508, 609.2911377, -760.0447998, 745.2609863

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723812, upper bound: 554.9723813
time: 1.22 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723812, upper bound: 554.9723813
time: 1.20 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.57 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.57
Output dim: 0, lower bound: -554.9712994, upper bound: 554.9737705
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.57
Output dim: 0, lower bound: -554.9712994, upper bound: 554.9737705
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.57
Output dim: 0, lower bound: -554.9712994, upper bound: 554.9737705
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.57
Output dim: 0, lower bound: -554.9712994, upper bound: 554.9737705
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 4.57
Output dim: 0, lower bound: -554.9788157, upper bound: 554.9759393
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 4.57
Output dim: 0, lower bound: -554.9786611, upper bound: 554.9758636
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.57
Output dim: 0, lower bound: -554.9723812, upper bound: 554.9723813
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.57
Output dim: 0, lower bound: -554.9723812, upper bound: 554.9723813

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -131.8264923, 415.5336609, -135.4273682, 427.6317749, -559.4582520, 550.9610596
1: -184.7457886, 418.9423218, -189.9990234, 430.8988953, -615.6446533, 608.9413452
2: -156.0319977, 463.5839844, -160.4833527, 476.7970886, -632.8289795, 624.0673218
3: -164.7066956, 594.9381104, -169.3506470, 612.0695801, -776.7761841, 764.2886353
4: -140.2548218, 544.5361328, -144.2209015, 559.9951782, -700.2499390, 688.7570190

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9772207
time: 1.22 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9772083
time: 1.18 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -131.8264923, 415.5336609, -141.9587555, 452.3752747, -584.2017822, 557.4924316
1: -184.7457886, 418.9423218, -199.8379517, 454.9771729, -639.7229614, 618.7802124
2: -156.0319977, 463.5839844, -168.7648926, 503.0252075, -659.0571289, 632.3486938
3: -164.7066956, 594.9381104, -178.0149994, 646.6646729, -811.3712769, 772.9530640
4: -140.2548218, 544.5361328, -151.5778503, 590.7100830, -730.9648438, 696.1139526

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9772207
time: 1.00 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9772083
time: 1.09 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -138.0580139, 439.1935120, -135.4273682, 427.6317749, -565.6898193, 574.6208496
1: -194.1572723, 441.8686218, -189.9990234, 430.8988953, -625.0559692, 631.8676758
2: -163.9659271, 488.6260071, -160.4833527, 476.7970886, -640.7630005, 649.1093750
3: -173.0027618, 627.9880371, -169.3506470, 612.0695801, -785.0722656, 797.3386230
4: -147.3025970, 573.9549561, -144.2209015, 559.9951782, -707.2977905, 718.1758423

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9692845, upper bound: 554.9722962
time: 1.12 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9688229, upper bound: 554.9714450
time: 0.96 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -138.0580139, 439.1935120, -141.9587555, 452.3752747, -590.4332886, 581.1522827
1: -194.1572723, 441.8686218, -199.8379517, 454.9771729, -649.1344604, 641.7065430
2: -163.9659271, 488.6260071, -168.7648926, 503.0252075, -666.9911499, 657.3907471
3: -173.0027618, 627.9880371, -178.0149994, 646.6646729, -819.6673584, 806.0030518
4: -147.3025970, 573.9549561, -151.5778503, 590.7100830, -738.0126953, 725.5327759

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9704521, upper bound: 554.9718567
time: 0.99 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9688229, upper bound: 554.9714450
time: 1.20 seconds

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: -133.0537109, 418.0853271, -146.2964935, 465.2480774, -598.3017578, 564.3818359
1: -185.2976227, 421.8851013, -205.7017822, 468.0694580, -653.3670654, 627.5867920
2: -156.6186371, 466.9302368, -173.7218781, 517.6021118, -674.2207642, 640.6520386
3: -165.4773560, 599.5919189, -183.3170929, 665.1445312, -830.6218262, 782.9089966
4: -140.9972992, 549.2918701, -156.0711975, 607.8920898, -748.8893433, 705.3630371

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9788157, upper bound: 554.9759393
time: 0.94 seconds

## Relational analysis of IS_A2_A1_A1_B2

### Relational analysis result of IS_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9788157, upper bound: 554.9759393
time: 1.06 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: -132.3233948, 415.1637573, -146.4843292, 465.8937683, -598.2171631, 561.6480713
1: -184.7430267, 419.2759399, -205.9705200, 468.7106323, -653.4536743, 625.2463989
2: -156.0207367, 464.0734558, -173.9482574, 518.3224487, -674.3432007, 638.0217285
3: -164.8735657, 595.7244873, -183.5567780, 666.0563354, -830.9298096, 779.2812500
4: -140.3941803, 546.0523071, -156.2746887, 608.7249146, -749.1190796, 702.3269653

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A1_A2_B1

### Relational analysis result of IS_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9758636
time: 1.21 seconds

## Relational analysis of IS_A2_A1_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9758636
time: 1.05 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -141.6680908, 449.4027405, -133.8571625, 422.3947754, -564.0628052, 583.2598877
1: -198.6617737, 452.4513550, -187.7098236, 425.7141418, -624.3759155, 640.1611938
2: -167.7134552, 500.6373596, -158.5439758, 471.1107178, -638.8239746, 659.1813354
3: -177.1431580, 643.5125732, -167.3309631, 604.6958618, -781.8388672, 810.8435059
4: -150.7537537, 588.8465576, -142.5009918, 553.3998413, -704.1535645, 731.3475342

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9704350, upper bound: 554.9708247
time: 1.26 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9699735
time: 1.08 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -141.6680908, 449.4027405, -140.3936615, 447.2259216, -588.8940430, 589.7963867
1: -198.6617737, 452.4513550, -197.5482941, 449.8397217, -648.5014648, 649.9995728
2: -167.7134552, 500.6373596, -166.8195648, 497.3952942, -665.1085815, 667.4569092
3: -177.1431580, 643.5125732, -175.9939575, 639.4146729, -816.5577393, 819.5065308
4: -150.7537537, 588.8465576, -149.8533325, 584.1387329, -734.8924561, 738.6998901

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708247, upper bound: 554.9704351
time: 1.04 seconds

## Relational analysis of IS_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9699735
time: 0.94 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.89 seconds
IS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9772207
IS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9772083
IS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9772207
IS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9772083
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 0, lower bound: -554.9692845, upper bound: 554.9722962
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 0, lower bound: -554.9688229, upper bound: 554.9714450
IS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 0, lower bound: -554.9704521, upper bound: 554.9718567
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 0, lower bound: -554.9688229, upper bound: 554.9714450
IS_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 0, lower bound: -554.9788157, upper bound: 554.9759393
IS_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 0, lower bound: -554.9788157, upper bound: 554.9759393
IS_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9758636
IS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9758636
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 0, lower bound: -554.9704350, upper bound: 554.9708247
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9699735
IS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 0, lower bound: -554.9708247, upper bound: 554.9704351
IS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 0, lower bound: -554.9699735, upper bound: 554.9699735

## BFS IS instance: IS_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -129.5068207, 407.2422180, -135.2338562, 426.9710083, -556.4778442, 542.4759521
1: -180.8375549, 410.8070984, -189.7218170, 430.2442627, -611.0816650, 600.5288696
2: -152.8676300, 454.5437622, -160.2490387, 476.0622559, -628.9298706, 614.7927246
3: -161.3594513, 583.3352661, -169.1035156, 611.1395874, -772.4989014, 752.4387817
4: -137.5561218, 533.8437500, -144.0109253, 559.1458130, -696.7018433, 677.8546753

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_A1_A1

### Relational analysis result of IS_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835770, upper bound: 554.9862391
time: 1.15 seconds

## Relational analysis of IS_A1_A1_B1_A1_A2

### Relational analysis result of IS_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835690, upper bound: 554.9844568
time: 0.86 seconds

## BFS IS instance: IS_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -129.1332550, 405.8225708, -135.4273682, 427.6317749, -556.7649536, 541.2499390
1: -180.8202820, 409.6738281, -189.9990234, 430.8988953, -611.7189941, 599.6728516
2: -152.7237396, 453.3137512, -160.4833527, 476.7970886, -629.5208130, 613.7971191
3: -161.2488556, 581.6445312, -169.3506470, 612.0695801, -773.3183594, 750.9951172
4: -137.3544464, 532.4979858, -144.2209015, 559.9951782, -697.3495483, 676.7188721

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_A2_A1

### Relational analysis result of IS_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835770, upper bound: 554.9863535
time: 0.91 seconds

## Relational analysis of IS_A1_A1_B1_A2_A2

### Relational analysis result of IS_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835752, upper bound: 554.9853167
time: 1.07 seconds

## BFS IS instance: IS_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -129.5068207, 407.2422180, -141.7680359, 451.7231750, -581.2299805, 549.0102539
1: -180.8375549, 410.8070984, -199.5653534, 454.3289795, -635.1663818, 610.3724365
2: -152.8676300, 454.5437622, -168.5351410, 502.2973938, -655.1650391, 623.0788574
3: -161.3594513, 583.3352661, -177.7719421, 645.7437744, -807.1030884, 761.1071777
4: -137.5561218, 533.8437500, -151.3714905, 589.8688965, -727.4249268, 685.2152100

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_A1_A1

### Relational analysis result of IS_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9772207
time: 1.16 seconds

## Relational analysis of IS_A1_A1_B2_A1_A2

### Relational analysis result of IS_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9776804, upper bound: 554.9757692
time: 1.13 seconds

## BFS IS instance: IS_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -129.1332550, 405.8225708, -141.9587555, 452.3752747, -581.5085449, 547.7811890
1: -180.8202820, 409.6738281, -199.8379517, 454.9771729, -635.7974854, 609.5117798
2: -152.7237396, 453.3137512, -168.7648926, 503.0252075, -655.7489624, 622.0786133
3: -161.2488556, 581.6445312, -178.0149994, 646.6646729, -807.9135132, 759.6595459
4: -137.3544464, 532.4979858, -151.5778503, 590.7100830, -728.0644531, 684.0758057

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9772083
time: 1.52 seconds

## Relational analysis of IS_A1_A1_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9776865, upper bound: 554.9766291
time: 1.12 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -137.8730774, 438.5633240, -133.1318359, 419.4592896, -557.3323975, 571.6950073
1: -193.8932495, 441.2402344, -186.1202240, 422.8794556, -616.7727051, 627.3604736
2: -163.7432556, 487.9223938, -157.3373566, 467.8631592, -631.6063843, 645.2597656
3: -172.7672424, 627.0946655, -166.0329590, 600.6323853, -773.3995361, 793.1275024
4: -147.1025238, 573.1432495, -141.5417938, 549.4505005, -696.5530396, 714.6850586

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9743585, upper bound: 554.9802873
time: 1.06 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9746938, upper bound: 554.9802873
time: 1.00 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -138.0580139, 439.1935120, -132.7345734, 417.8988953, -555.9569092, 571.9281006
1: -194.1572723, 441.8686218, -186.0755157, 421.6268921, -615.7841187, 627.9441528
2: -163.9659271, 488.6260071, -157.1767426, 466.5088501, -630.4747314, 645.8027344
3: -173.0027618, 627.9880371, -165.8930664, 598.7429199, -771.7456055, 793.8811035
4: -147.3025970, 573.9549561, -141.3210754, 547.9201050, -695.2227173, 715.2760010

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9743762, upper bound: 554.9801327
time: 1.27 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9747116, upper bound: 554.9801327
time: 0.78 seconds

## BFS IS instance: IS_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -135.6968231, 430.7904663, -141.7680359, 451.7231750, -587.4199829, 572.5584717
1: -190.1970978, 433.5797424, -199.5653534, 454.3289795, -644.5259399, 633.1450806
2: -160.7775116, 479.4364319, -168.5351410, 502.2973938, -663.0748901, 647.9715576
3: -169.6004028, 616.1744995, -177.7719421, 645.7437744, -815.3441772, 793.9464111
4: -144.5901337, 563.0349731, -151.3714905, 589.8688965, -734.4589844, 714.4064941

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B2_A1_A1

### Relational analysis result of IS_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9703201, upper bound: 554.9718567
time: 0.91 seconds

## Relational analysis of IS_A1_A2_B2_A1_A2

### Relational analysis result of IS_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9703201, upper bound: 554.9718567
time: 1.41 seconds

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -135.3625336, 429.4769592, -141.9587555, 452.3752747, -587.7377930, 571.4356689
1: -190.2272034, 432.5570984, -199.8379517, 454.9771729, -645.2043457, 632.3950195
2: -160.6483917, 478.3566895, -168.7648926, 503.0252075, -663.6735840, 647.1215210
3: -169.5371857, 614.6412354, -178.0149994, 646.6646729, -816.2018433, 792.6562500
4: -144.3938904, 561.8070068, -151.5778503, 590.7100830, -735.1039429, 713.3848267

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B2_A2_A1

### Relational analysis result of IS_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684876, upper bound: 554.9714450
time: 1.17 seconds

## Relational analysis of IS_A1_A2_B2_A2_A2

### Relational analysis result of IS_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9688229, upper bound: 554.9714450
time: 1.37 seconds

## BFS IS instance: IS_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -133.0537109, 418.0853271, -133.6610260, 421.7243652, -554.7780762, 551.7463379
1: -185.2976227, 421.8851013, -187.4285889, 425.0522461, -610.3497925, 609.3136597
2: -156.6186371, 466.9302368, -158.3062286, 470.3680115, -626.9866333, 625.2364502
3: -165.4773560, 599.5919189, -167.0803375, 603.7554321, -769.2326660, 766.6722412
4: -140.9972992, 549.2918701, -142.2880402, 552.5407715, -693.5380859, 691.5798340

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B1_A1

### Relational analysis result of IS_A2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9785626, upper bound: 554.9759393
time: 1.15 seconds

## Relational analysis of IS_A2_A1_A1_B1_A2

### Relational analysis result of IS_A2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9788157, upper bound: 554.9758444
time: 1.29 seconds

## BFS IS instance: IS_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -133.0537109, 418.0853271, -140.2021179, 446.5710144, -579.6246338, 558.2874756
1: -185.2976227, 421.8851013, -197.2743225, 449.1886902, -634.4863281, 619.1594238
2: -156.6186371, 466.9302368, -166.5886993, 496.6646423, -653.2832642, 633.5189209
3: -165.4773560, 599.5919189, -175.7496948, 638.4895020, -803.9667969, 775.3416138
4: -140.9972992, 549.2918701, -149.6459351, 583.2936401, -724.2909546, 698.9376831

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B2_A1

### Relational analysis result of IS_A2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9785626, upper bound: 554.9759393
time: 1.37 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2

### Relational analysis result of IS_A2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9788157, upper bound: 554.9758444
time: 0.97 seconds

## BFS IS instance: IS_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -132.3233948, 415.1637573, -133.8571625, 422.3947754, -554.7181396, 549.0208740
1: -184.7430267, 419.2759399, -187.7098236, 425.7141418, -610.4571533, 606.9857788
2: -156.0207367, 464.0734558, -158.5439758, 471.1107178, -627.1313477, 622.6174316
3: -164.8735657, 595.7244873, -167.3309631, 604.6958618, -769.5691528, 763.0554199
4: -140.3941803, 546.0523071, -142.5009918, 553.3998413, -693.7940063, 688.5532227

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A2_B1_A1

### Relational analysis result of IS_A2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9779077, upper bound: 554.9758636
time: 1.26 seconds

## Relational analysis of IS_A2_A1_A2_B1_A2

### Relational analysis result of IS_A2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9786611, upper bound: 554.9758621
time: 1.02 seconds

## BFS IS instance: IS_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -132.3233948, 415.1637573, -140.3936615, 447.2259216, -579.5493164, 555.5574341
1: -184.7430267, 419.2759399, -197.5482941, 449.8397217, -634.5827637, 616.8240967
2: -156.0207367, 464.0734558, -166.8195648, 497.3952942, -653.4160156, 630.8930054
3: -164.8735657, 595.7244873, -175.9939575, 639.4146729, -804.2880859, 771.7183838
4: -140.3941803, 546.0523071, -149.8533325, 584.1387329, -724.5328979, 695.9056396

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A2_B2_A1

### Relational analysis result of IS_A2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9779077, upper bound: 554.9758636
time: 1.07 seconds

## Relational analysis of IS_A2_A1_A2_B2_A2

### Relational analysis result of IS_A2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9786611, upper bound: 554.9758621
time: 1.04 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -141.4674835, 448.7102051, -131.5740356, 414.2812195, -555.7487183, 580.2842407
1: -198.3762207, 451.7659607, -183.8404694, 417.7760925, -616.1523438, 635.6064453
2: -167.4718933, 499.8679199, -155.4048615, 462.2446899, -629.7164917, 655.2726440
3: -176.8873901, 642.5333252, -164.0231171, 593.3699951, -770.2572021, 806.5564575
4: -150.5352173, 587.9585571, -139.8298798, 542.9313354, -693.4665527, 727.7883301

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9759393, upper bound: 554.9785626
time: 1.10 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9758444, upper bound: 554.9788157
time: 1.22 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -141.6680908, 449.4027405, -131.1634369, 412.6549377, -554.3229370, 580.5661621
1: -198.6617737, 452.4513550, -183.7887573, 416.4399719, -615.1017456, 636.2401123
2: -167.7134552, 500.6373596, -155.2374115, 460.8222351, -628.5356445, 655.8747559
3: -177.1431580, 643.5125732, -163.8725586, 591.3778687, -768.5209961, 807.3851318
4: -150.7537537, 588.8465576, -139.6007080, 541.3403931, -692.0941162, 728.4472046

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9758635, upper bound: 554.9779077
time: 0.88 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9758621, upper bound: 554.9786611
time: 0.76 seconds

## BFS IS instance: IS_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -139.2046967, 440.9418640, -140.2021179, 446.5710144, -585.7755737, 581.1439819
1: -194.6376648, 444.0958252, -197.2743225, 449.1886902, -643.8263550, 641.3701172
2: -164.4753876, 491.3377380, -166.5886993, 496.6646423, -661.1398926, 657.9263916
3: -173.6896820, 631.5960083, -175.7496948, 638.4895020, -812.1790771, 807.3457031
4: -147.9734955, 577.8323364, -149.6459351, 583.2936401, -731.2671509, 727.4781494

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9706352, upper bound: 554.9689711
time: 1.27 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9695233, upper bound: 554.9690090
time: 1.25 seconds

## BFS IS instance: IS_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -138.9383545, 439.6115417, -140.3936615, 447.2259216, -586.1643066, 580.0051880
1: -194.6855316, 443.0742798, -197.5482941, 449.8397217, -644.5252686, 640.6224365
2: -164.3618774, 490.2503662, -166.8195648, 497.3952942, -661.7570190, 657.0699463
3: -173.6382446, 630.0073853, -175.9939575, 639.4146729, -813.0528564, 806.0012817
4: -147.8123169, 576.6052856, -149.8533325, 584.1387329, -731.9509277, 726.4586182

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9697840, upper bound: 554.9686342
time: 1.19 seconds

## Relational analysis of IS_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9686721, upper bound: 554.9686721
time: 0.86 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 7.42 seconds
IS_A1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9835770, upper bound: 554.9862391
IS_A1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9835690, upper bound: 554.9844568
IS_A1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9835770, upper bound: 554.9863535
IS_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9835752, upper bound: 554.9853167
IS_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9772207
IS_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9776804, upper bound: 554.9757692
IS_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9772083
IS_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9776865, upper bound: 554.9766291
IS_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9743585, upper bound: 554.9802873
IS_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9746938, upper bound: 554.9802873
IS_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9743762, upper bound: 554.9801327
IS_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9747116, upper bound: 554.9801327
IS_A1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9703201, upper bound: 554.9718567
IS_A1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9703201, upper bound: 554.9718567
IS_A1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9684876, upper bound: 554.9714450
IS_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9688229, upper bound: 554.9714450
IS_A2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9785626, upper bound: 554.9759393
IS_A2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9788157, upper bound: 554.9758444
IS_A2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9785626, upper bound: 554.9759393
IS_A2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9788157, upper bound: 554.9758444
IS_A2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9779077, upper bound: 554.9758636
IS_A2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9786611, upper bound: 554.9758621
IS_A2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9779077, upper bound: 554.9758636
IS_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9786611, upper bound: 554.9758621
IS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9759393, upper bound: 554.9785626
IS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9758444, upper bound: 554.9788157
IS_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9758635, upper bound: 554.9779077
IS_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9758621, upper bound: 554.9786611
IS_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9706352, upper bound: 554.9689711
IS_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9695233, upper bound: 554.9690090
IS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9697840, upper bound: 554.9686342
IS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 7.42
Output dim: 0, lower bound: -554.9686721, upper bound: 554.9686721

## BFS IS instance: IS_A1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -110.0623703, 344.9439087, -135.2338562, 426.9710083, -537.0333862, 480.1777649
1: -153.4287872, 348.4817505, -189.7218170, 430.2442627, -583.6730347, 538.2035522
2: -129.8651886, 385.4705811, -160.2490387, 476.0622559, -605.9273682, 545.7195435
3: -136.9499359, 495.2588196, -169.1035156, 611.1395874, -748.0895386, 664.3623047
4: -117.0508270, 453.0469971, -144.0109253, 559.1458130, -676.1965332, 597.0578613

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835690, upper bound: 554.9837034
time: 0.97 seconds

## Relational analysis of IS_A1_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835690, upper bound: 554.9844568
time: 1.33 seconds

## BFS IS instance: IS_A1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -125.5976486, 394.2683716, -135.2338562, 426.9710083, -552.5686646, 529.5021362
1: -175.2224121, 398.0465393, -189.7218170, 430.2442627, -605.4664917, 587.7683716
2: -148.1581421, 440.4619141, -160.2490387, 476.0622559, -624.2203979, 600.7106934
3: -156.3659668, 565.0202026, -169.1035156, 611.1395874, -767.5055542, 734.1237183
4: -133.3429871, 517.2592163, -144.0109253, 559.1458130, -692.4887695, 661.2701416

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835690, upper bound: 554.9837034
time: 0.97 seconds

## Relational analysis of IS_A1_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835690, upper bound: 554.9844568
time: 1.08 seconds

## BFS IS instance: IS_A1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -109.1081619, 341.8386841, -135.4273682, 427.6317749, -536.7399292, 477.2660522
1: -152.6162720, 345.6217041, -189.9990234, 430.8988953, -583.5150146, 535.6207275
2: -129.0530243, 382.3387451, -160.4833527, 476.7970886, -605.8500977, 542.8220825
3: -136.1234894, 491.2404480, -169.3506470, 612.0695801, -748.1930542, 660.5910645
4: -116.2431488, 449.5137024, -144.2209015, 559.9951782, -676.2383423, 593.7346191

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835752, upper bound: 554.9845633
time: 1.05 seconds

## Relational analysis of IS_A1_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835752, upper bound: 554.9853167
time: 1.52 seconds

## BFS IS instance: IS_A1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -125.1999207, 392.7516785, -135.4273682, 427.6317749, -552.8316040, 528.1789551
1: -175.1900024, 396.7728882, -189.9990234, 430.8988953, -606.0886841, 586.7719116
2: -148.0010223, 439.1036682, -160.4833527, 476.7970886, -624.7980957, 599.5870361
3: -156.2403412, 563.2280273, -169.3506470, 612.0695801, -768.3098145, 732.5786743
4: -133.1271515, 515.8400269, -144.2209015, 559.9951782, -693.1222534, 660.0609131

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835752, upper bound: 554.9845633
time: 1.58 seconds

## Relational analysis of IS_A1_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835752, upper bound: 554.9853167
time: 0.96 seconds

## BFS IS instance: IS_A1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -110.0623703, 344.9439087, -141.7680359, 451.7231750, -561.7855225, 486.7119446
1: -153.4287872, 348.4817505, -199.5653534, 454.3289795, -607.7576904, 548.0471191
2: -129.8651886, 385.4705811, -168.5351410, 502.2973938, -632.1624756, 554.0056152
3: -136.9499359, 495.2588196, -177.7719421, 645.7437744, -782.6936646, 673.0307617
4: -117.0508270, 453.0469971, -151.3714905, 589.8688965, -706.9196167, 604.4183960

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_A1_A1_A1

### Relational analysis result of IS_A1_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9706758, upper bound: 554.9753221
time: 1.53 seconds

## Relational analysis of IS_A1_A1_B2_A1_A1_A2

### Relational analysis result of IS_A1_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9769364
time: 1.14 seconds

## BFS IS instance: IS_A1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -125.5976486, 394.2683716, -141.7680359, 451.7231750, -577.3208008, 536.0363770
1: -175.2224121, 398.0465393, -199.5653534, 454.3289795, -629.5511475, 597.6118774
2: -148.1581421, 440.4619141, -168.5351410, 502.2973938, -650.4555664, 608.9968872
3: -156.3659668, 565.0202026, -177.7719421, 645.7437744, -802.1097412, 742.7921143
4: -133.3429871, 517.2592163, -151.3714905, 589.8688965, -723.2118530, 668.6306763

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_A1_A2_A1

### Relational analysis result of IS_A1_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9776804, upper bound: 554.9757692
time: 1.12 seconds

## Relational analysis of IS_A1_A1_B2_A1_A2_A2

### Relational analysis result of IS_A1_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9751554, upper bound: 554.9733990
time: 1.07 seconds

## BFS IS instance: IS_A1_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -109.1081619, 341.8386841, -141.9587555, 452.3752747, -561.4834595, 483.7974243
1: -152.6162720, 345.6217041, -199.8379517, 454.9771729, -607.5934448, 545.4596558
2: -129.0530243, 382.3387451, -168.7648926, 503.0252075, -632.0782471, 551.1036377
3: -136.1234894, 491.2404480, -178.0149994, 646.6646729, -782.7881470, 669.2554321
4: -116.2431488, 449.5137024, -151.5778503, 590.7100830, -706.9532471, 601.0914917

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_A2_A1_A1

### Relational analysis result of IS_A1_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9660558, upper bound: 554.9703359
time: 1.27 seconds

## Relational analysis of IS_A1_A1_B2_A2_A1_A2

### Relational analysis result of IS_A1_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9769826
time: 1.26 seconds

## BFS IS instance: IS_A1_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -125.1999207, 392.7516785, -141.9587555, 452.3752747, -577.5751953, 534.7102661
1: -175.1900024, 396.7728882, -199.8379517, 454.9771729, -630.1671753, 596.6108398
2: -148.0010223, 439.1036682, -168.7648926, 503.0252075, -651.0262451, 607.8685303
3: -156.2403412, 563.2280273, -178.0149994, 646.6646729, -802.9049683, 741.2430420
4: -133.1271515, 515.8400269, -151.5778503, 590.7100830, -723.8372192, 667.4178467

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_A2_A2_A1

### Relational analysis result of IS_A1_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9776865, upper bound: 554.9766291
time: 1.08 seconds

## Relational analysis of IS_A1_A1_B2_A2_A2_A2

### Relational analysis result of IS_A1_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9751550, upper bound: 554.9736587
time: 1.04 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -119.2834320, 379.1982727, -133.1318359, 419.4592896, -538.7427368, 512.3300781
1: -167.7207489, 381.6195679, -186.1202240, 422.8794556, -590.6002197, 567.7398071
2: -141.7736969, 422.0183105, -157.3373566, 467.8631592, -609.6367798, 579.3556519
3: -149.4496460, 542.9333496, -166.0329590, 600.6323853, -750.0820312, 708.9662476
4: -127.4716797, 495.9111633, -141.5417938, 549.4505005, -676.9221802, 637.4529419

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9743585, upper bound: 554.9800341
time: 1.40 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9743585, upper bound: 554.9802873
time: 0.91 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -133.3740997, 423.5095825, -133.1318359, 419.4592896, -552.8332520, 556.6414185
1: -187.4670715, 426.3372498, -186.1202240, 422.8794556, -610.3465576, 612.4574585
2: -158.3555756, 471.5888062, -157.3373566, 467.8631592, -626.2187500, 628.9261475
3: -167.0513153, 605.7098999, -166.0329590, 600.6323853, -767.6837158, 771.7426758
4: -142.2764435, 553.8504028, -141.5417938, 549.4505005, -691.7269287, 695.3922119

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9746938, upper bound: 554.9800341
time: 1.12 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_B2

### Relational analysis result of IS_A1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9746938, upper bound: 554.9802873
time: 1.35 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -119.4554749, 379.7881470, -132.7345734, 417.8988953, -537.3543701, 512.5226440
1: -167.9652405, 382.2072449, -186.0755157, 421.6268921, -589.5921021, 568.2826538
2: -141.9803619, 422.6773682, -157.1767426, 466.5088501, -608.4891968, 579.8541260
3: -149.6687622, 543.7709351, -165.8930664, 598.7429199, -748.4116211, 709.6640015
4: -127.6578827, 496.6718750, -141.3210754, 547.9201050, -675.5780029, 637.9929199

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9743762, upper bound: 554.9793793
time: 0.94 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9743762, upper bound: 554.9801327
time: 1.21 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -133.5896454, 424.2452698, -132.7345734, 417.8988953, -551.4885254, 556.9798584
1: -187.7752838, 427.0671387, -186.0755157, 421.6268921, -609.4020996, 613.1426392
2: -158.6146698, 472.4032898, -157.1767426, 466.5088501, -625.1234741, 629.5800171
3: -167.3257751, 606.7480469, -165.8930664, 598.7429199, -766.0687256, 772.6411133
4: -142.5086670, 554.7945557, -141.3210754, 547.9201050, -690.4287720, 696.1156006

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9747116, upper bound: 554.9793793
time: 1.15 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9747116, upper bound: 554.9801327
time: 1.13 seconds

## BFS IS instance: IS_A1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -117.6247635, 373.0070496, -141.7680359, 451.7231750, -569.3479614, 514.7750854
1: -164.7247620, 375.5840454, -199.5653534, 454.3289795, -619.0536499, 575.1494141
2: -139.3993225, 415.2898560, -168.5351410, 502.2973938, -641.6966553, 583.8248291
3: -146.9249573, 534.2611084, -177.7719421, 645.7437744, -792.6687012, 712.0330811
4: -125.5041580, 487.9191589, -151.3714905, 589.8688965, -715.3729858, 639.2905884

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B2_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9649636, upper bound: 554.9713346
time: 1.48 seconds

## Relational analysis of IS_A1_A2_B2_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9696973, upper bound: 554.9717040
time: 0.99 seconds

## BFS IS instance: IS_A1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -131.3785248, 416.3287048, -141.7680359, 451.7231750, -583.1016846, 558.0967407
1: -184.0023193, 419.3077393, -199.5653534, 454.3289795, -638.3310547, 618.8731079
2: -155.5814667, 463.7756653, -168.5351410, 502.2973938, -657.8788452, 632.3106689
3: -164.0947113, 595.6860352, -177.7719421, 645.7437744, -809.8385010, 773.4580078
4: -139.9371033, 544.5258789, -151.3714905, 589.8688965, -729.8060303, 695.8973389

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B2_A1_A2_A1

### Relational analysis result of IS_A1_A2_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9484027, upper bound: 554.9660836
time: 1.40 seconds

## Relational analysis of IS_A1_A2_B2_A1_A2_A2

### Relational analysis result of IS_A1_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9704386, upper bound: 554.9717040
time: 1.24 seconds

## BFS IS instance: IS_A1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -116.7944489, 370.2088318, -141.9587555, 452.3752747, -569.1697388, 512.1676025
1: -164.0859680, 373.0873718, -199.8379517, 454.9771729, -619.0631104, 572.9252930
2: -138.7025909, 412.5715637, -168.7648926, 503.0252075, -641.7277832, 581.3364258
3: -146.2432098, 530.6113281, -178.0149994, 646.6646729, -792.9078979, 708.6263428
4: -124.7826920, 484.7663574, -151.5778503, 590.7100830, -715.4927368, 636.3442383

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B2_A2_A1_A1

### Relational analysis result of IS_A1_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9605569, upper bound: 554.9697921
time: 1.49 seconds

## Relational analysis of IS_A1_A2_B2_A2_A1_A2

### Relational analysis result of IS_A1_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9680810, upper bound: 554.9712858
time: 1.05 seconds

## BFS IS instance: IS_A1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -130.9656372, 414.7893982, -141.9587555, 452.3752747, -583.3409424, 556.7481689
1: -183.9457092, 418.0145264, -199.8379517, 454.9771729, -638.9228516, 617.8524170
2: -155.3790741, 462.4112854, -168.7648926, 503.0252075, -658.4042969, 631.1761475
3: -163.9487000, 593.8269043, -178.0149994, 646.6646729, -810.6133423, 771.8419189
4: -139.6714478, 543.0300903, -151.5778503, 590.7100830, -730.3815308, 694.6079102

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B2_A2_A2_A1

### Relational analysis result of IS_A1_A2_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9257557, upper bound: 554.9430342
time: 0.97 seconds

## Relational analysis of IS_A1_A2_B2_A2_A2_A2

### Relational analysis result of IS_A1_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9688229, upper bound: 554.9712858
time: 1.11 seconds

## BFS IS instance: IS_A2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -114.2197342, 358.4015198, -133.6610260, 421.7243652, -535.9440918, 492.0625000
1: -158.8167572, 361.8819275, -187.4285889, 425.0522461, -583.8690186, 549.3104248
2: -134.3860016, 400.5310059, -158.3062286, 470.3680115, -604.7540283, 558.8372192
3: -141.8735352, 514.9166260, -167.0803375, 603.7554321, -745.6289062, 681.9969482
4: -121.1714172, 471.5576477, -142.2880402, 552.5407715, -673.7121582, 613.8457031

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9844512, upper bound: 554.9837787
time: 1.12 seconds

## Relational analysis of IS_A2_A1_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9844512, upper bound: 554.9845320
time: 1.14 seconds

## BFS IS instance: IS_A2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -129.2108612, 405.3808899, -133.6610260, 421.7243652, -550.9352417, 539.0418701
1: -179.7688599, 409.2690125, -187.4285889, 425.0522461, -604.8211060, 596.6976318
2: -152.0035248, 453.0260010, -158.3062286, 470.3680115, -622.3715210, 611.3322144
3: -160.5584259, 581.4934692, -167.0803375, 603.7554321, -764.3138428, 748.5737915
4: -136.8525543, 532.8260498, -142.2880402, 552.5407715, -689.3933105, 675.1140137

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9847044, upper bound: 554.9837787
time: 1.02 seconds

## Relational analysis of IS_A2_A1_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9847044, upper bound: 554.9845320
time: 0.87 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -114.2197342, 358.4015198, -140.2021179, 446.5710144, -560.7906494, 498.6036377
1: -158.8167572, 361.8819275, -197.2743225, 449.1886902, -608.0054321, 559.1562500
2: -134.3860016, 400.5310059, -166.5886993, 496.6646423, -631.0505981, 567.1196899
3: -141.8735352, 514.9166260, -175.7496948, 638.4895020, -780.3630371, 690.6663208
4: -121.1714172, 471.5576477, -149.6459351, 583.2936401, -704.4650879, 621.2036133

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_A1_B2_A1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9785626, upper bound: 554.9734020
time: 1.24 seconds

## Relational analysis of IS_A2_A1_A1_B2_A1_B2

### Relational analysis result of IS_A2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9785626, upper bound: 554.9742564
time: 0.87 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -129.2108612, 405.3808899, -140.2021179, 446.5710144, -575.7818604, 545.5830078
1: -179.7688599, 409.2690125, -197.2743225, 449.1886902, -628.9575195, 606.5433350
2: -152.0035248, 453.0260010, -166.5886993, 496.6646423, -648.6681519, 619.6146851
3: -160.5584259, 581.4934692, -175.7496948, 638.4895020, -799.0479126, 757.2431641
4: -136.8525543, 532.8260498, -149.6459351, 583.2936401, -720.1461792, 682.4718628

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_A1_B2_A2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9788157, upper bound: 554.9733071
time: 1.21 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9788157, upper bound: 554.9743703
time: 1.13 seconds

## BFS IS instance: IS_A2_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -112.9218140, 354.0248108, -133.8571625, 422.3947754, -535.3165894, 487.8819580
1: -157.4883118, 357.7160645, -187.7098236, 425.7141418, -583.2024536, 545.4259033
2: -133.1354065, 396.0085754, -158.5439758, 471.1107178, -604.2460327, 554.5525513
3: -140.5787964, 508.9474487, -167.3309631, 604.6958618, -745.2746582, 676.2784424
4: -119.9698105, 466.3082886, -142.5009918, 553.3998413, -673.3696289, 608.8092041

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A2_B1_A1_B1

### Relational analysis result of IS_A2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9837964, upper bound: 554.9837964
time: 0.89 seconds

## Relational analysis of IS_A2_A1_A2_B1_A1_B2

### Relational analysis result of IS_A2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9837964, upper bound: 554.9845498
time: 1.04 seconds

## BFS IS instance: IS_A2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -128.4108582, 402.1971436, -133.8571625, 422.3947754, -550.8056641, 536.0542603
1: -179.1265717, 406.3817444, -187.7098236, 425.7141418, -604.8406982, 594.0915527
2: -151.3296204, 449.8663025, -158.5439758, 471.1107178, -622.4400635, 608.4102783
3: -159.8763733, 577.3211060, -167.3309631, 604.6958618, -764.5720215, 744.6520996
4: -136.1824951, 529.3269043, -142.5009918, 553.3998413, -689.5823364, 671.8278809

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A2_B1_A2_B1

### Relational analysis result of IS_A2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9845498, upper bound: 554.9837964
time: 1.33 seconds

## Relational analysis of IS_A2_A1_A2_B1_A2_B2

### Relational analysis result of IS_A2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9845498, upper bound: 554.9845498
time: 1.21 seconds

## BFS IS instance: IS_A2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -112.9218140, 354.0248108, -140.3936615, 447.2259216, -560.1477051, 494.4184570
1: -157.4883118, 357.7160645, -197.5482941, 449.8397217, -607.3280029, 555.2642822
2: -133.1354065, 396.0085754, -166.8195648, 497.3952942, -630.5307007, 562.8281250
3: -140.5787964, 508.9474487, -175.9939575, 639.4146729, -779.9934692, 684.9414062
4: -119.9698105, 466.3082886, -149.8533325, 584.1387329, -704.1084595, 616.1615601

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_A2_B2_A1_B1

### Relational analysis result of IS_A2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9779077, upper bound: 554.9733263
time: 0.97 seconds

## Relational analysis of IS_A2_A1_A2_B2_A1_B2

### Relational analysis result of IS_A2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9779077, upper bound: 554.9734842
time: 1.06 seconds

## BFS IS instance: IS_A2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -128.4108582, 402.1971436, -140.3936615, 447.2259216, -575.6367798, 542.5908203
1: -179.1265717, 406.3817444, -197.5482941, 449.8397217, -628.9663086, 603.9298706
2: -151.3296204, 449.8663025, -166.8195648, 497.3952942, -648.7247925, 616.6858521
3: -159.8763733, 577.3211060, -175.9939575, 639.4146729, -799.2909546, 753.3150024
4: -136.1824951, 529.3269043, -149.8533325, 584.1387329, -720.3212280, 679.1802368

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_A2_B2_A2_B1

### Relational analysis result of IS_A2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9786611, upper bound: 554.9733249
time: 1.25 seconds

## Relational analysis of IS_A2_A1_A2_B2_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9786611, upper bound: 554.9740684
time: 1.15 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -141.4674835, 448.7102051, -112.2070389, 352.1691589, -493.6365967, 560.9171753
1: -198.3762207, 451.7659607, -156.5572357, 355.6697998, -554.0460205, 608.3231812
2: -167.4718933, 499.8679199, -132.5099792, 393.4328308, -560.9046631, 632.3778687
3: -176.8873901, 642.5333252, -139.7209473, 505.5868225, -682.4741211, 782.2542725
4: -150.5352173, 587.9585571, -119.4167709, 462.4442444, -612.9793701, 707.3751831

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9757909, upper bound: 554.9756047
time: 0.81 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9759393, upper bound: 554.9776884
time: 0.89 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9759393, upper bound: 554.9784219
time: 1.06 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -141.4674835, 448.7102051, -127.8707733, 401.9470825, -543.4145508, 576.5809326
1: -198.3762207, 451.7659607, -178.5119781, 405.6609192, -604.0371094, 630.2778931
2: -167.4718933, 499.8679199, -150.9453430, 448.8361511, -616.3079224, 650.8132324
3: -176.8873901, 642.5333252, -159.2827759, 575.9182739, -752.8055420, 801.8161011
4: -150.5352173, 587.9585571, -135.8300781, 527.1223145, -677.6575317, 723.7885132

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9757420, upper bound: 554.9760254
time: 1.08 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9725081, upper bound: 554.9752622
time: 0.99 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -141.6680908, 449.4027405, -111.1928711, 348.8748779, -490.5429688, 560.5955811
1: -198.6617737, 452.4513550, -155.6725311, 352.6167297, -551.2785034, 608.1239014
2: -167.7134552, 500.6373596, -131.6437378, 390.1016541, -557.8150635, 632.2810669
3: -177.1431580, 643.5125732, -138.8206177, 501.2689209, -678.4121094, 782.3331909
4: -150.7537537, 588.8465576, -118.5547943, 458.6274109, -609.3811646, 707.4013672

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9757633, upper bound: 554.9750098
time: 1.10 seconds

## Relational analysis of IS_A2_A2_B1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9758636, upper bound: 554.9776884
time: 1.07 seconds

## Relational analysis of IS_A2_A2_B1_B2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9758636, upper bound: 554.9778877
time: 1.02 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -141.6680908, 449.4027405, -127.4541168, 400.3129272, -541.9810181, 576.8568726
1: -198.6617737, 452.4513550, -178.4719696, 404.2841492, -602.9459229, 630.9233398
2: -167.7134552, 500.6373596, -150.7817993, 447.3833313, -615.0965576, 651.4189453
3: -177.1431580, 643.5125732, -159.1417694, 573.9628296, -751.1059570, 802.6543579
4: -150.7537537, 588.8465576, -135.6071777, 525.5836792, -676.3373413, 724.4537354

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9757633, upper bound: 554.9759255
time: 1.24 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9721609, upper bound: 554.9749239
time: 1.10 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -139.2046967, 440.9418640, -138.5193024, 440.9569397, -580.1614380, 579.4608765
1: -194.6376648, 444.0958252, -194.8796692, 443.6114807, -638.2491455, 638.9754639
2: -164.4753876, 491.3377380, -164.5748291, 490.4887695, -654.9640503, 655.9124756
3: -173.6896820, 631.5960083, -173.6247406, 630.5462646, -804.2358398, 805.2205811
4: -147.9734955, 577.8323364, -147.8566589, 576.0578613, -724.0313721, 725.6889648

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9706352, upper bound: 554.9660484
time: 1.28 seconds

## Relational analysis of IS_A2_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9706352, upper bound: 554.9689485
time: 1.52 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -138.6201782, 438.9952393, -159.4038544, 511.5563354, -650.1762695, 598.3989868
1: -193.8005066, 442.1783752, -224.5616455, 512.8098755, -706.6103516, 666.7399902
2: -163.7713623, 489.2127686, -189.6703949, 567.2188721, -730.9902344, 678.8829346
3: -172.9471588, 628.8599243, -200.0126801, 730.5223389, -903.4694214, 828.8725586
4: -147.3494263, 575.3656006, -170.3424530, 667.0652466, -814.4146729, 745.7078857

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9695233, upper bound: 554.9690090
time: 0.97 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9695233, upper bound: 554.9690090
time: 1.29 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -138.9383545, 439.6115417, -138.7092896, 441.6079712, -580.5463257, 578.3208008
1: -194.6855316, 443.0742798, -195.1514740, 444.2580261, -638.9435425, 638.2257080
2: -164.3618774, 490.2503662, -164.8037720, 491.2145386, -655.5764160, 655.0541382
3: -173.6382446, 630.0073853, -173.8670349, 631.4655762, -805.1037598, 803.8743286
4: -147.8123169, 576.6052856, -148.0623322, 576.8972778, -724.7094727, 724.6674805

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9697840, upper bound: 554.9657115
time: 1.04 seconds

## Relational analysis of IS_A2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9697840, upper bound: 554.9686342
time: 1.04 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -138.3569183, 437.6795654, -159.6058502, 512.2250977, -650.5819702, 597.2854004
1: -193.8535614, 441.1699524, -224.8504791, 513.4815063, -707.3350830, 666.0204468
2: -163.6616516, 488.1397705, -189.9137115, 567.9705200, -731.6322021, 678.0534668
3: -172.9005737, 627.2926025, -200.2703094, 731.4700317, -904.3705444, 827.5629272
4: -147.1918945, 574.1625366, -170.5604706, 667.9329834, -815.1248169, 744.7229004

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9686721, upper bound: 554.9686721
time: 0.93 seconds

## Relational analysis of IS_A2_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9686721, upper bound: 554.9686721
time: 1.05 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 8.57 seconds
IS_A1_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9835690, upper bound: 554.9837034
IS_A1_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9835690, upper bound: 554.9844568
IS_A1_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9835690, upper bound: 554.9837034
IS_A1_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9835690, upper bound: 554.9844568
IS_A1_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9835752, upper bound: 554.9845633
IS_A1_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9835752, upper bound: 554.9853167
IS_A1_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9835752, upper bound: 554.9845633
IS_A1_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9835752, upper bound: 554.9853167
IS_A1_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9706758, upper bound: 554.9753221
IS_A1_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9769364
IS_A1_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9776804, upper bound: 554.9757692
IS_A1_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9751554, upper bound: 554.9733990
IS_A1_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9660558, upper bound: 554.9703359
IS_A1_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9769826
IS_A1_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9776865, upper bound: 554.9766291
IS_A1_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9751550, upper bound: 554.9736587
IS_A1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9743585, upper bound: 554.9800341
IS_A1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9743585, upper bound: 554.9802873
IS_A1_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9746938, upper bound: 554.9800341
IS_A1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9746938, upper bound: 554.9802873
IS_A1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9743762, upper bound: 554.9793793
IS_A1_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9743762, upper bound: 554.9801327
IS_A1_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9747116, upper bound: 554.9793793
IS_A1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9747116, upper bound: 554.9801327
IS_A1_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9649636, upper bound: 554.9713346
IS_A1_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9696973, upper bound: 554.9717040
IS_A1_A2_B2_A1_A2_A1, status: Status.VERIFIED, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9484027, upper bound: 554.9660836
IS_A1_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9704386, upper bound: 554.9717040
IS_A1_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9605569, upper bound: 554.9697921
IS_A1_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9680810, upper bound: 554.9712858
IS_A1_A2_B2_A2_A2_A1, status: Status.VERIFIED, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9257557, upper bound: 554.9430342
IS_A1_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9688229, upper bound: 554.9712858
IS_A2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9844512, upper bound: 554.9837787
IS_A2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9844512, upper bound: 554.9845320
IS_A2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9847044, upper bound: 554.9837787
IS_A2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9847044, upper bound: 554.9845320
IS_A2_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9785626, upper bound: 554.9734020
IS_A2_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9785626, upper bound: 554.9742564
IS_A2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9788157, upper bound: 554.9733071
IS_A2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9788157, upper bound: 554.9743703
IS_A2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9837964, upper bound: 554.9837964
IS_A2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9837964, upper bound: 554.9845498
IS_A2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9845498, upper bound: 554.9837964
IS_A2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9845498, upper bound: 554.9845498
IS_A2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9779077, upper bound: 554.9733263
IS_A2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9779077, upper bound: 554.9734842
IS_A2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9786611, upper bound: 554.9733249
IS_A2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9786611, upper bound: 554.9740684
IS_A2_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9759393, upper bound: 554.9776884
IS_A2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9759393, upper bound: 554.9784219
IS_A2_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9757420, upper bound: 554.9760254
IS_A2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9725081, upper bound: 554.9752622
IS_A2_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9758636, upper bound: 554.9776884
IS_A2_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9758636, upper bound: 554.9778877
IS_A2_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9757633, upper bound: 554.9759255
IS_A2_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9721609, upper bound: 554.9749239
IS_A2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9706352, upper bound: 554.9660484
IS_A2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9706352, upper bound: 554.9689485
IS_A2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9695233, upper bound: 554.9690090
IS_A2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9695233, upper bound: 554.9690090
IS_A2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9697840, upper bound: 554.9657115
IS_A2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9697840, upper bound: 554.9686342
IS_A2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9686721, upper bound: 554.9686721
IS_A2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 8.57
Output dim: 0, lower bound: -554.9686721, upper bound: 554.9686721

## BFS IS instance: IS_A1_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -110.0623703, 344.9439087, -115.2127457, 362.8926392, -472.9549866, 460.1566467
1: -153.4287872, 348.4817505, -161.5377655, 366.1794128, -519.6082153, 510.0194702
2: -129.8651886, 385.4705811, -136.5985107, 405.0575867, -534.9226685, 522.0689697
3: -136.9499359, 495.2588196, -144.0032959, 520.5358276, -657.4855957, 639.2620850
4: -117.0508270, 453.0469971, -122.9200363, 476.1209106, -593.1716919, 575.9669800

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9766868, upper bound: 554.9849462
time: 1.63 seconds

## Relational analysis of IS_A1_A1_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835088, upper bound: 554.9850180
time: 0.86 seconds

## BFS IS instance: IS_A1_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -110.0623703, 344.9439087, -131.3125458, 413.9220886, -523.9844360, 476.2564697
1: -153.4287872, 348.4817505, -184.1101990, 417.3779907, -570.8067017, 532.5918579
2: -129.8651886, 385.4705811, -155.5483551, 461.8611145, -591.7262573, 541.0188599
3: -136.9499359, 495.2588196, -164.1098785, 592.5761108, -729.5260620, 659.3687134
4: -117.0508270, 453.0469971, -139.8022003, 542.4275513, -659.4783325, 592.8491821

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9766868, upper bound: 554.9855366
time: 1.03 seconds

## Relational analysis of IS_A1_A1_B1_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835088, upper bound: 554.9857714
time: 0.87 seconds

## BFS IS instance: IS_A1_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -125.5976486, 394.2683716, -115.2127457, 362.8926392, -488.4902344, 509.4811096
1: -175.2224121, 398.0465393, -161.5377655, 366.1794128, -541.4016724, 559.5842896
2: -148.1581421, 440.4619141, -136.5985107, 405.0575867, -553.2156982, 577.0601807
3: -156.3659668, 565.0202026, -144.0032959, 520.5358276, -676.9017944, 709.0234985
4: -133.3429871, 517.2592163, -122.9200363, 476.1209106, -609.4638672, 640.1791992

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835505, upper bound: 554.9837034
time: 1.00 seconds

## Relational analysis of IS_A1_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835505, upper bound: 554.9837034
time: 1.17 seconds

## BFS IS instance: IS_A1_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -125.5976486, 394.2683716, -131.3125458, 413.9220886, -539.5197144, 525.5809326
1: -175.2224121, 398.0465393, -184.1101990, 417.3779907, -592.6001587, 582.1566162
2: -148.1581421, 440.4619141, -155.5483551, 461.8611145, -610.0192871, 596.0100708
3: -156.3659668, 565.0202026, -164.1098785, 592.5761108, -748.9420776, 729.1300659
4: -133.3429871, 517.2592163, -139.8022003, 542.4275513, -675.7705078, 657.0614014

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9810440, upper bound: 554.9820866
time: 1.01 seconds

## Relational analysis of IS_A1_A1_B1_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_A1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835505, upper bound: 554.9837034
time: 1.25 seconds

## Relational analysis of IS_A1_A1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835505, upper bound: 554.9837034
time: 1.13 seconds

## BFS IS instance: IS_A1_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -109.1081619, 341.8386841, -115.3944626, 363.5161438, -472.6242981, 457.2331238
1: -152.6162720, 345.6217041, -161.7974091, 366.7948303, -519.4111328, 507.4190979
2: -129.0530243, 382.3387451, -136.8181305, 405.7499084, -534.8029175, 519.1568604
3: -136.1234894, 491.2404480, -144.2350464, 521.4130859, -657.5365601, 635.4754639
4: -116.2431488, 449.5137024, -123.1167984, 476.9207153, -593.1638794, 572.6304932

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A2_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9751476, upper bound: 554.9847138
time: 0.95 seconds

## Relational analysis of IS_A1_A1_B1_A2_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835088, upper bound: 554.9854927
time: 1.05 seconds

## BFS IS instance: IS_A1_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -109.1081619, 341.8386841, -131.5360107, 414.6815186, -523.7896729, 473.3746948
1: -152.6162720, 345.6217041, -184.4300995, 418.1284180, -570.7445679, 530.0518188
2: -129.0530243, 382.3387451, -155.8180084, 462.7027588, -591.7557983, 538.1567383
3: -136.1234894, 491.2404480, -164.3947754, 593.6577148, -729.7811890, 655.6352539
4: -116.2431488, 449.5137024, -140.0432434, 543.4032593, -659.6464233, 589.5569458

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A2_A1_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9751476, upper bound: 554.9848392
time: 0.99 seconds

## Relational analysis of IS_A1_A1_B1_A2_A1_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835088, upper bound: 554.9861271
time: 1.17 seconds

## BFS IS instance: IS_A1_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -125.1999207, 392.7516785, -115.3944626, 363.5161438, -488.7160645, 508.1461487
1: -175.1900024, 396.7728882, -161.7974091, 366.7948303, -541.9848633, 558.5703125
2: -148.0010223, 439.1036682, -136.8181305, 405.7499084, -553.7509155, 575.9218140
3: -156.2403412, 563.2280273, -144.2350464, 521.4130859, -677.6533813, 707.4630737
4: -133.1271515, 515.8400269, -123.1167984, 476.9207153, -610.0478516, 638.9567871

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_A1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835567, upper bound: 554.9845633
time: 1.09 seconds

## Relational analysis of IS_A1_A1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835567, upper bound: 554.9845633
time: 1.30 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 7.32 seconds
IS_A1_A1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.32
Output dim: 0, lower bound: -554.9766868, upper bound: 554.9849462
IS_A1_A1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.32
Output dim: 0, lower bound: -554.9835088, upper bound: 554.9850180
IS_A1_A1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.32
Output dim: 0, lower bound: -554.9766868, upper bound: 554.9855366
IS_A1_A1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.32
Output dim: 0, lower bound: -554.9835088, upper bound: 554.9857714
IS_A1_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.32
Output dim: 0, lower bound: -554.9835505, upper bound: 554.9837034
IS_A1_A1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.32
Output dim: 0, lower bound: -554.9835505, upper bound: 554.9837034
IS_A1_A1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.32
Output dim: 0, lower bound: -554.9835505, upper bound: 554.9837034
IS_A1_A1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.32
Output dim: 0, lower bound: -554.9835505, upper bound: 554.9837034
IS_A1_A1_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.32
Output dim: 0, lower bound: -554.9751476, upper bound: 554.9847138
IS_A1_A1_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.32
Output dim: 0, lower bound: -554.9835088, upper bound: 554.9854927
IS_A1_A1_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.32
Output dim: 0, lower bound: -554.9751476, upper bound: 554.9848392
IS_A1_A1_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.32
Output dim: 0, lower bound: -554.9835088, upper bound: 554.9861271
IS_A1_A1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.32
Output dim: 0, lower bound: -554.9835567, upper bound: 554.9845633
IS_A1_A1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.32
Output dim: 0, lower bound: -554.9835567, upper bound: 554.9845633
IS_A1_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9835752, upper bound: 554.9853167
IS_A1_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9706758, upper bound: 554.9753221
IS_A1_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9769364
IS_A1_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9776804, upper bound: 554.9757692
IS_A1_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9751554, upper bound: 554.9733990
IS_A1_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9660558, upper bound: 554.9703359
IS_A1_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9776884, upper bound: 554.9769826
IS_A1_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9776865, upper bound: 554.9766291
IS_A1_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9751550, upper bound: 554.9736587
IS_A1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9743585, upper bound: 554.9800341
IS_A1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9743585, upper bound: 554.9802873
IS_A1_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9746938, upper bound: 554.9800341
IS_A1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9746938, upper bound: 554.9802873
IS_A1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9743762, upper bound: 554.9793793
IS_A1_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9743762, upper bound: 554.9801327
IS_A1_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9747116, upper bound: 554.9793793
IS_A1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9747116, upper bound: 554.9801327
IS_A1_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9649636, upper bound: 554.9713346
IS_A1_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9696973, upper bound: 554.9717040
IS_A1_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9704386, upper bound: 554.9717040
IS_A1_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9605569, upper bound: 554.9697921
IS_A1_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9680810, upper bound: 554.9712858
IS_A1_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9688229, upper bound: 554.9712858
IS_A2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9844512, upper bound: 554.9837787
IS_A2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9844512, upper bound: 554.9845320
IS_A2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9847044, upper bound: 554.9837787
IS_A2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9847044, upper bound: 554.9845320
IS_A2_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9785626, upper bound: 554.9734020
IS_A2_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9785626, upper bound: 554.9742564
IS_A2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9788157, upper bound: 554.9733071
IS_A2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9788157, upper bound: 554.9743703
IS_A2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9837964, upper bound: 554.9837964
IS_A2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9837964, upper bound: 554.9845498
IS_A2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9845498, upper bound: 554.9837964
IS_A2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9845498, upper bound: 554.9845498
IS_A2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9779077, upper bound: 554.9733263
IS_A2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9779077, upper bound: 554.9734842
IS_A2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9786611, upper bound: 554.9733249
IS_A2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9786611, upper bound: 554.9740684
IS_A2_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9759393, upper bound: 554.9776884
IS_A2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9759393, upper bound: 554.9784219
IS_A2_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9757420, upper bound: 554.9760254
IS_A2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9725081, upper bound: 554.9752622
IS_A2_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9758636, upper bound: 554.9776884
IS_A2_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9758636, upper bound: 554.9778877
IS_A2_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9757633, upper bound: 554.9759255
IS_A2_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9721609, upper bound: 554.9749239
IS_A2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9706352, upper bound: 554.9660484
IS_A2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9706352, upper bound: 554.9689485
IS_A2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9695233, upper bound: 554.9690090
IS_A2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9695233, upper bound: 554.9690090
IS_A2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9697840, upper bound: 554.9657115
IS_A2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9697840, upper bound: 554.9686342
IS_A2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9686721, upper bound: 554.9686721
IS_A2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.32
Output dim: 0, lower bound: -554.9686721, upper bound: 554.9686721
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=619.6494140625
rel_dist={0: [-554.9911089992902, 554.9911089992902]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9842007, upper bound: 554.9863407
time: 0.87 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9853709, upper bound: 554.9853709
time: 1.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.40 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.40
Output dim: 0, lower bound: -554.9842007, upper bound: 554.9863407
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.40
Output dim: 0, lower bound: -554.9853709, upper bound: 554.9853709

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -144.4209442, 458.6982422, -148.1843414, 471.4650574, -615.8859863, 606.8823853
1: -202.9677582, 461.6006775, -208.4494476, 474.2722168, -677.2398071, 670.0501099
2: -171.4153137, 510.5334778, -176.0516663, 524.4273071, -695.8425293, 686.5851440
3: -180.9070740, 655.8257446, -185.7463837, 673.9080200, -854.8150024, 841.5721436
4: -154.0161438, 599.6799927, -158.1413727, 615.8510742, -769.8671875, 757.8212891

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9779812, upper bound: 554.9779347
time: 1.19 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9711060, upper bound: 554.9737668
time: 1.31 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -147.6497955, 467.7235718, -143.9573364, 457.5959473, -605.2456665, 611.6809082
1: -206.9226227, 471.0274048, -202.2762451, 460.4558411, -667.3782959, 673.3035889
2: -174.7036133, 521.2171631, -170.8094482, 509.2548523, -683.9583130, 692.0266113
3: -184.5714264, 669.6788330, -180.2973328, 654.4045410, -838.9759521, 849.9761353
4: -157.0737762, 613.0461426, -153.4935913, 598.1820679, -755.2557983, 766.5395508

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9794071, upper bound: 554.9779016
time: 1.18 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723811, upper bound: 554.9723812
time: 1.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.47 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 4.47
Output dim: 0, lower bound: -554.9779812, upper bound: 554.9779347
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 4.47
Output dim: 0, lower bound: -554.9711060, upper bound: 554.9737668
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 4.47
Output dim: 0, lower bound: -554.9794071, upper bound: 554.9779016
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 4.47
Output dim: 0, lower bound: -554.9723811, upper bound: 554.9723812

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -131.8264923, 415.5336609, -146.9334869, 467.3382263, -599.1647339, 562.4671631
1: -184.7457886, 418.9423218, -206.6879730, 470.1475525, -654.8933105, 625.6301270
2: -156.0319977, 463.5839844, -174.5671082, 519.8568115, -675.8886719, 638.1510010
3: -164.7066956, 594.9381104, -184.1661987, 668.0075684, -832.7141724, 779.1042480
4: -140.2548218, 544.5361328, -156.8076019, 610.4512329, -750.7059937, 701.3437500

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9711060, upper bound: 554.9737668
time: 1.29 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9711060, upper bound: 554.9737668
time: 0.98 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -138.0580139, 439.1935120, -146.2898865, 465.6584473, -603.7164307, 585.4833984
1: -194.1572723, 441.8686218, -205.8313904, 468.4003906, -662.5576782, 647.7000122
2: -163.9659271, 488.6260071, -173.8343811, 517.9138184, -681.8797607, 662.4601440
3: -173.0027618, 627.9880371, -183.3922729, 665.6176147, -838.6203003, 811.3802490
4: -147.3025970, 573.9549561, -156.1433868, 608.1997070, -755.5023193, 730.0982666

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9711060, upper bound: 554.9737668
time: 1.07 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9711060, upper bound: 554.9737668
time: 0.99 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -134.9726562, 424.7419739, -142.7309265, 453.5307617, -588.5034180, 567.4729004
1: -188.6135559, 428.4528198, -200.5504761, 456.3942566, -645.0077515, 629.0032959
2: -159.2701263, 474.2145996, -169.3571930, 504.7537842, -664.0238037, 643.5717773
3: -168.2837219, 608.8666992, -178.7485199, 648.5986938, -816.8824463, 787.6152344
4: -143.2449951, 557.9903564, -152.1866150, 592.8648071, -736.1097412, 710.1770020

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723811, upper bound: 554.9723812
time: 1.33 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723811, upper bound: 554.9723812
time: 1.18 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -141.6680908, 449.4027405, -142.0035248, 451.5908508, -593.2589111, 591.4062500
1: -198.6617737, 452.4513550, -199.5712433, 454.3890686, -653.0508423, 652.0225830
2: -167.7134552, 500.6373596, -168.5178528, 502.5252686, -670.2385254, 669.1550903
3: -177.1431580, 643.5125732, -177.8650513, 645.8344116, -822.9775391, 821.3776245
4: -150.7537537, 588.8465576, -151.4294281, 590.2748413, -741.0285645, 740.2759399

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723811, upper bound: 554.9723812
time: 1.17 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723811, upper bound: 554.9723812
time: 1.18 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.96 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.96
Output dim: 0, lower bound: -554.9711060, upper bound: 554.9737668
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.96
Output dim: 0, lower bound: -554.9711060, upper bound: 554.9737668
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.96
Output dim: 0, lower bound: -554.9711060, upper bound: 554.9737668
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.96
Output dim: 0, lower bound: -554.9711060, upper bound: 554.9737668
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.96
Output dim: 0, lower bound: -554.9723811, upper bound: 554.9723812
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.96
Output dim: 0, lower bound: -554.9723811, upper bound: 554.9723812
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.96
Output dim: 0, lower bound: -554.9723811, upper bound: 554.9723812
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.96
Output dim: 0, lower bound: -554.9723811, upper bound: 554.9723812

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -131.8264923, 415.5336609, -135.4273682, 427.6317749, -559.4582520, 550.9610596
1: -184.7457886, 418.9423218, -189.9990234, 430.8988953, -615.6446533, 608.9413452
2: -156.0319977, 463.5839844, -160.4833527, 476.7970886, -632.8289795, 624.0673218
3: -164.7066956, 594.9381104, -169.3506470, 612.0695801, -776.7761841, 764.2886353
4: -140.2548218, 544.5361328, -144.2209015, 559.9951782, -700.2499390, 688.7570190

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9756507
time: 1.02 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9755471
time: 1.01 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -131.8264923, 415.5336609, -141.9587555, 452.3752747, -584.2017822, 557.4924316
1: -184.7457886, 418.9423218, -199.8379517, 454.9771729, -639.7229614, 618.7802124
2: -156.0319977, 463.5839844, -168.7648926, 503.0252075, -659.0571289, 632.3486938
3: -164.7066956, 594.9381104, -178.0149994, 646.6646729, -811.3712769, 772.9530640
4: -140.2548218, 544.5361328, -151.5778503, 590.7100830, -730.9648438, 696.1139526

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9756507
time: 1.02 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9755471
time: 1.68 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -138.0580139, 439.1935120, -135.4273682, 427.6317749, -565.6898193, 574.6208496
1: -194.1572723, 441.8686218, -189.9990234, 430.8988953, -625.0559692, 631.8676758
2: -163.9659271, 488.6260071, -160.4833527, 476.7970886, -640.7630005, 649.1093750
3: -173.0027618, 627.9880371, -169.3506470, 612.0695801, -785.0722656, 797.3386230
4: -147.3025970, 573.9549561, -144.2209015, 559.9951782, -707.2977905, 718.1758423

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9689560, upper bound: 554.9718825
time: 1.37 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684885, upper bound: 554.9709713
time: 1.24 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -138.0580139, 439.1935120, -141.9587555, 452.3752747, -590.4332886, 581.1522827
1: -194.1572723, 441.8686218, -199.8379517, 454.9771729, -649.1344604, 641.7065430
2: -163.9659271, 488.6260071, -168.7648926, 503.0252075, -666.9911499, 657.3907471
3: -173.0027618, 627.9880371, -178.0149994, 646.6646729, -819.6673584, 806.0030518
4: -147.3025970, 573.9549561, -151.5778503, 590.7100830, -738.0126953, 725.5327759

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9702253, upper bound: 554.9716902
time: 1.08 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684885, upper bound: 554.9709713
time: 1.20 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -134.9726562, 424.7419739, -131.1896515, 413.5289917, -548.5016479, 555.9316406
1: -188.6135559, 428.4528198, -183.8209991, 416.9697571, -605.5831909, 612.2738037
2: -159.2701263, 474.2145996, -155.2470551, 461.4402771, -620.7102661, 629.4616699
3: -168.2837219, 608.8666992, -163.9000244, 592.2707520, -760.5544434, 772.7665405
4: -143.2449951, 557.9903564, -139.5769653, 542.1956787, -685.4406738, 697.5671997

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9786191, upper bound: 554.9758371
time: 1.39 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9784923, upper bound: 554.9756416
time: 1.67 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -134.9726562, 424.7419739, -137.6571350, 438.2232361, -573.1958618, 562.3991089
1: -188.6135559, 428.4528198, -193.5435486, 440.8949585, -629.5085449, 621.9963379
2: -159.2701263, 474.2145996, -163.4193726, 487.5547791, -646.8248291, 637.6339722
3: -168.2837219, 608.8666992, -172.4597015, 626.7738647, -795.0576172, 781.3263550
4: -143.2449951, 557.9903564, -146.8401947, 572.6825562, -715.9274902, 704.8304443

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9786191, upper bound: 554.9758371
time: 1.50 seconds

## Relational analysis of IS_A2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9784923, upper bound: 554.9756416
time: 1.19 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -141.6680908, 449.4027405, -131.1896515, 413.5289917, -555.1969604, 580.5924072
1: -198.6617737, 452.4513550, -183.8209991, 416.9697571, -615.6315308, 636.2722168
2: -167.7134552, 500.6373596, -155.2470551, 461.4402771, -629.1536865, 655.8843994
3: -177.1431580, 643.5125732, -163.9000244, 592.2707520, -769.4138794, 807.4125366
4: -150.7537537, 588.8465576, -139.5769653, 542.1956787, -692.9494629, 728.4234619

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9704324, upper bound: 554.9708176
time: 1.05 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699661, upper bound: 554.9699662
time: 1.11 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -141.6680908, 449.4027405, -137.6571350, 438.2232361, -579.8911743, 587.0598755
1: -198.6617737, 452.4513550, -193.5435486, 440.8949585, -639.5567627, 645.9948730
2: -167.7134552, 500.6373596, -163.4193726, 487.5547791, -655.2681274, 664.0564575
3: -177.1431580, 643.5125732, -172.4597015, 626.7738647, -803.9168701, 815.9722900
4: -150.7537537, 588.8465576, -146.8401947, 572.6825562, -723.4362793, 735.6867065

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708176, upper bound: 554.9704325
time: 1.07 seconds

## Relational analysis of IS_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699661, upper bound: 554.9699662
time: 0.97 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.98 seconds
IS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9756507
IS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9755471
IS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9756507
IS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9755471
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -554.9689560, upper bound: 554.9718825
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -554.9684885, upper bound: 554.9709713
IS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -554.9702253, upper bound: 554.9716902
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -554.9684885, upper bound: 554.9709713
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -554.9786191, upper bound: 554.9758371
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -554.9784923, upper bound: 554.9756416
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -554.9786191, upper bound: 554.9758371
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -554.9784923, upper bound: 554.9756416
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -554.9704324, upper bound: 554.9708176
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -554.9699661, upper bound: 554.9699662
IS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -554.9708176, upper bound: 554.9704325
IS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -554.9699661, upper bound: 554.9699662

## BFS IS instance: IS_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -129.5068207, 407.2422180, -134.6154175, 424.8596191, -554.3664551, 541.8576660
1: -180.8375549, 410.8070984, -188.8361359, 428.1529846, -608.9905396, 599.6431274
2: -152.8676300, 454.5437622, -159.5000305, 473.7153625, -626.5830078, 614.0438232
3: -161.3594513, 583.3352661, -168.3139191, 608.1669312, -769.5262451, 751.6491699
4: -137.5561218, 533.8437500, -143.3395996, 556.4315796, -693.9876709, 677.1833496

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9857245
time: 1.14 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9857245
time: 0.96 seconds

## BFS IS instance: IS_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -129.1332550, 405.8225708, -135.4273682, 427.6317749, -556.7649536, 541.2499390
1: -180.8202820, 409.6738281, -189.9990234, 430.8988953, -611.7189941, 599.6728516
2: -152.7237396, 453.3137512, -160.4833527, 476.7970886, -629.5208130, 613.7971191
3: -161.2488556, 581.6445312, -169.3506470, 612.0695801, -773.3183594, 750.9951172
4: -137.3544464, 532.4979858, -144.2209015, 559.9951782, -697.3495483, 676.7188721

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9857245
time: 1.07 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9857245
time: 0.99 seconds

## BFS IS instance: IS_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -129.5068207, 407.2422180, -141.1639557, 449.6545410, -579.1613770, 548.4061279
1: -180.8375549, 410.8070984, -198.7019043, 452.2731323, -633.1105957, 609.5090332
2: -152.8676300, 454.5437622, -167.8073425, 499.9893799, -652.8569946, 622.3510742
3: -161.3594513, 583.3352661, -177.0014648, 642.8223877, -804.1817017, 760.3366699
4: -137.5561218, 533.8437500, -150.7167816, 587.2012329, -724.7572632, 684.5603638

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9739138
time: 1.15 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9756507
time: 1.07 seconds

## BFS IS instance: IS_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -129.1332550, 405.8225708, -141.9587555, 452.3752747, -581.5085449, 547.7811890
1: -180.8202820, 409.6738281, -199.8379517, 454.9771729, -635.7974854, 609.5117798
2: -152.7237396, 453.3137512, -168.7648926, 503.0252075, -655.7489624, 622.0786133
3: -161.2488556, 581.6445312, -178.0149994, 646.6646729, -807.9135132, 759.6595459
4: -137.3544464, 532.4979858, -151.5778503, 590.7100830, -728.0644531, 684.0758057

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9755471
time: 1.19 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9755471
time: 1.06 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -137.2875824, 436.5627747, -133.1318359, 419.4592896, -556.7468872, 569.6945801
1: -193.0572510, 439.2462158, -186.1202240, 422.8794556, -615.9367065, 625.3664551
2: -163.0380554, 485.6901245, -157.3373566, 467.8631592, -630.9011230, 643.0274658
3: -172.0207977, 624.2588501, -166.0329590, 600.6323853, -772.6531372, 790.2916870
4: -146.4680939, 570.5666504, -141.5417938, 549.4505005, -695.9185791, 712.1084595

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9747058, upper bound: 554.9797114
time: 1.01 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9747058, upper bound: 554.9797114
time: 1.18 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -138.0580139, 439.1935120, -132.7345734, 417.8988953, -555.9569092, 571.9281006
1: -194.1572723, 441.8686218, -186.0755157, 421.6268921, -615.7841187, 627.9441528
2: -163.9659271, 488.6260071, -157.1767426, 466.5088501, -630.4747314, 645.8027344
3: -173.0027618, 627.9880371, -165.8930664, 598.7429199, -771.7456055, 793.8811035
4: -147.3025970, 573.9549561, -141.3210754, 547.9201050, -695.2227173, 715.2760010

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9747058, upper bound: 554.9797114
time: 1.13 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9747058, upper bound: 554.9797114
time: 1.29 seconds

## BFS IS instance: IS_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -135.6968231, 430.7904663, -141.1639557, 449.6545410, -585.3511963, 571.9544067
1: -190.1970978, 433.5797424, -198.7019043, 452.2731323, -642.4701538, 632.2816162
2: -160.7775116, 479.4364319, -167.8073425, 499.9893799, -660.7669067, 647.2437744
3: -169.6004028, 616.1744995, -177.0014648, 642.8223877, -812.4227295, 793.1758423
4: -144.5901337, 563.0349731, -150.7167816, 587.2012329, -731.7913208, 713.7517090

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684885, upper bound: 554.9709713
time: 1.19 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684885, upper bound: 554.9709713
time: 0.77 seconds

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -135.3625336, 429.4769592, -141.9587555, 452.3752747, -587.7377930, 571.4356689
1: -190.2272034, 432.5570984, -199.8379517, 454.9771729, -645.2043457, 632.3950195
2: -160.6483917, 478.3566895, -168.7648926, 503.0252075, -663.6735840, 647.1215210
3: -169.5371857, 614.6412354, -178.0149994, 646.6646729, -816.2018433, 792.6562500
4: -144.3938904, 561.8070068, -151.5778503, 590.7100830, -735.1039429, 713.3848267

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684885, upper bound: 554.9709713
time: 1.47 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684885, upper bound: 554.9709713
time: 1.21 seconds

## BFS IS instance: IS_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -133.0537109, 418.0853271, -130.3686523, 410.7269287, -543.7806396, 548.4539795
1: -185.2976227, 421.8851013, -182.6427002, 414.2008362, -599.4984131, 604.5277100
2: -156.6186371, 466.9302368, -154.2516174, 458.3363953, -614.9550171, 621.1818237
3: -165.4773560, 599.5919189, -162.8505402, 588.3248291, -753.8021851, 762.4424438
4: -140.9972992, 549.2918701, -138.6846619, 538.6027222, -679.5999756, 687.9765625

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9847058, upper bound: 554.9836222
time: 1.00 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9847058, upper bound: 554.9842424
time: 1.29 seconds

## BFS IS instance: IS_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -132.3233948, 415.1637573, -131.1896515, 413.5289917, -545.8524170, 546.3533936
1: -184.7430267, 419.2759399, -183.8209991, 416.9697571, -601.7127686, 603.0968018
2: -156.0207367, 464.0734558, -155.2470551, 461.4402771, -617.4609985, 619.3204956
3: -164.8735657, 595.7244873, -163.9000244, 592.2707520, -757.1442261, 759.6243286
4: -140.3941803, 546.0523071, -139.5769653, 542.1956787, -682.5898438, 685.6291504

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9845512, upper bound: 554.9845512
time: 1.26 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9845512, upper bound: 554.9845512
time: 0.93 seconds

## BFS IS instance: IS_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -133.0537109, 418.0853271, -136.8545990, 435.4744568, -568.5280762, 554.9399414
1: -185.2976227, 421.8851013, -192.3957977, 438.1668091, -623.4644165, 614.2808228
2: -156.6186371, 466.9302368, -162.4523621, 484.4935608, -641.1121826, 629.3825684
3: -165.4773560, 599.5919189, -171.4361572, 622.8916016, -788.3689575, 771.0280762
4: -140.9972992, 549.2918701, -145.9712067, 569.1371460, -710.1344604, 695.2630615

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9786191, upper bound: 554.9733778
time: 1.18 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9786191, upper bound: 554.9743703
time: 1.36 seconds

## BFS IS instance: IS_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -132.3233948, 415.1637573, -137.6571350, 438.2232361, -570.5466309, 552.8209229
1: -184.7430267, 419.2759399, -193.5435486, 440.8949585, -625.6380005, 612.8194580
2: -156.0207367, 464.0734558, -163.4193726, 487.5547791, -643.5755005, 627.4927368
3: -164.8735657, 595.7244873, -172.4597015, 626.7738647, -791.6472168, 768.1841431
4: -140.3941803, 546.0523071, -146.8401947, 572.6825562, -713.0767212, 692.8923950

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9784923, upper bound: 554.9756416
time: 0.93 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9784923, upper bound: 554.9756416
time: 1.23 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -140.8572083, 446.5912781, -128.9752045, 405.6633911, -546.5205688, 575.5664673
1: -197.5076294, 449.6712646, -180.0332184, 409.2827454, -606.7904053, 629.7044678
2: -166.7368164, 497.5155334, -152.1817017, 452.8391113, -619.5759277, 649.6972656
3: -176.1089325, 639.5375977, -160.6666718, 581.2777100, -757.3866577, 800.2042847
4: -149.8696289, 585.2446289, -136.9699707, 532.0159302, -681.8854980, 722.2145386

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9756506, upper bound: 554.9772056
time: 1.13 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9756507, upper bound: 554.9784350
time: 1.16 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -141.6680908, 449.4027405, -128.4959412, 403.7906494, -545.4586792, 577.8986816
1: -198.6617737, 452.4513550, -179.9012604, 407.6918030, -606.3535156, 632.3526001
2: -167.7134552, 500.6373596, -151.9429016, 451.1640015, -618.8774414, 652.5800781
3: -177.1431580, 643.5125732, -160.4413147, 578.9739990, -756.1171265, 803.9538574
4: -150.7537537, 588.8465576, -136.6783600, 530.1739502, -680.9276733, 725.5248413

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9756416, upper bound: 554.9784923
time: 1.10 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9756416, upper bound: 554.9784923
time: 1.29 seconds

## BFS IS instance: IS_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -139.2046967, 440.9418640, -136.8545990, 435.4744568, -574.6790161, 577.7963867
1: -194.6376648, 444.0958252, -192.3957977, 438.1668091, -632.8044434, 636.4916382
2: -164.4753876, 491.3377380, -162.4523621, 484.4935608, -648.9688110, 653.7901001
3: -173.6896820, 631.5960083, -171.4361572, 622.8916016, -796.5811768, 803.0321655
4: -147.9734955, 577.8323364, -145.9712067, 569.1371460, -717.1106567, 723.8035278

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708176, upper bound: 554.9678801
time: 5.70 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708176, upper bound: 554.9692813
time: 1.01 seconds

## BFS IS instance: IS_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -138.9383545, 439.6115417, -137.6571350, 438.2232361, -577.1616211, 577.2686768
1: -194.6855316, 443.0742798, -193.5435486, 440.8949585, -635.5805054, 636.6177979
2: -164.3618774, 490.2503662, -163.4193726, 487.5547791, -651.9166260, 653.6696777
3: -173.6382446, 630.0073853, -172.4597015, 626.7738647, -800.4119873, 802.4670410
4: -147.8123169, 576.6052856, -146.8401947, 572.6825562, -720.4948120, 723.4454346

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699661, upper bound: 554.9699662
time: 1.06 seconds

## Relational analysis of IS_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699661, upper bound: 554.9699662
time: 1.21 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.52 seconds
IS_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9857245
IS_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9857245
IS_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9857245
IS_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9857245
IS_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9739138
IS_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9756507
IS_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9755471
IS_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9755471
IS_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9747058, upper bound: 554.9797114
IS_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9747058, upper bound: 554.9797114
IS_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9747058, upper bound: 554.9797114
IS_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9747058, upper bound: 554.9797114
IS_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9684885, upper bound: 554.9709713
IS_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9684885, upper bound: 554.9709713
IS_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9684885, upper bound: 554.9709713
IS_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9684885, upper bound: 554.9709713
IS_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9847058, upper bound: 554.9836222
IS_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9847058, upper bound: 554.9842424
IS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9845512, upper bound: 554.9845512
IS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9845512, upper bound: 554.9845512
IS_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9786191, upper bound: 554.9733778
IS_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9786191, upper bound: 554.9743703
IS_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9784923, upper bound: 554.9756416
IS_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9784923, upper bound: 554.9756416
IS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9756506, upper bound: 554.9772056
IS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9756507, upper bound: 554.9784350
IS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9756416, upper bound: 554.9784923
IS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9756416, upper bound: 554.9784923
IS_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9708176, upper bound: 554.9678801
IS_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9708176, upper bound: 554.9692813
IS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9699661, upper bound: 554.9699662
IS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -554.9699661, upper bound: 554.9699662

## BFS IS instance: IS_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -129.5068207, 407.2422180, -133.1318359, 419.4592896, -548.9661255, 540.3739624
1: -180.8375549, 410.8070984, -186.1202240, 422.8794556, -603.7170410, 596.9273071
2: -152.8676300, 454.5437622, -157.3373566, 467.8631592, -620.7307739, 611.8811035
3: -161.3594513, 583.3352661, -166.0329590, 600.6323853, -761.9916992, 749.3681030
4: -137.5561218, 533.8437500, -141.5417938, 549.4505005, -687.0065918, 675.3854980

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9849517
time: 1.20 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9857507
time: 0.96 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -129.5068207, 407.2422180, -132.7345734, 417.8988953, -547.4057007, 539.9767456
1: -180.8375549, 410.8070984, -186.0755157, 421.6268921, -602.4643555, 596.8826294
2: -152.8676300, 454.5437622, -157.1767426, 466.5088501, -619.3764648, 611.7205200
3: -161.3594513, 583.3352661, -165.8930664, 598.7429199, -760.1022339, 749.2283325
4: -137.5561218, 533.8437500, -141.3210754, 547.9201050, -685.4761963, 675.1647949

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9849517
time: 0.94 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9857507
time: 1.18 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -129.1332550, 405.8225708, -133.1318359, 419.4592896, -548.5925293, 538.9542847
1: -180.8202820, 409.6738281, -186.1202240, 422.8794556, -603.6997070, 595.7940674
2: -152.7237396, 453.3137512, -157.3373566, 467.8631592, -620.5869141, 610.6511230
3: -161.2488556, 581.6445312, -166.0329590, 600.6323853, -761.8812256, 747.6773682
4: -137.3544464, 532.4979858, -141.5417938, 549.4505005, -686.8049316, 674.0397949

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9849565
time: 0.99 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9857245
time: 1.20 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -129.1332550, 405.8225708, -132.7345734, 417.8988953, -547.0321045, 538.5570679
1: -180.8202820, 409.6738281, -186.0755157, 421.6268921, -602.4470825, 595.7493286
2: -152.7237396, 453.3137512, -157.1767426, 466.5088501, -619.2325439, 610.4904785
3: -161.2488556, 581.6445312, -165.8930664, 598.7429199, -759.9917603, 747.5375977
4: -137.3544464, 532.4979858, -141.3210754, 547.9201050, -685.2745361, 673.8190918

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9849565
time: 1.17 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9857245
time: 1.06 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -129.5068207, 407.2422180, -137.2875824, 436.5627747, -566.0695801, 544.5297852
1: -180.8375549, 410.8070984, -193.0572510, 439.2462158, -620.0837402, 603.8643188
2: -152.8676300, 454.5437622, -163.0380554, 485.6901245, -638.5577393, 617.5817871
3: -161.3594513, 583.3352661, -172.0207977, 624.2588501, -785.6181641, 755.3560791
4: -137.5561218, 533.8437500, -146.4680939, 570.5666504, -708.1228027, 680.3118286

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9739138
time: 0.79 seconds

## Relational analysis of IS_A1_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9739138
time: 1.29 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -129.5068207, 407.2422180, -140.8154144, 446.4977417, -576.0045776, 548.0576172
1: -180.8375549, 410.8070984, -197.4614410, 449.5711365, -630.4085693, 608.2684326
2: -152.8676300, 454.5437622, -166.6932220, 497.4051819, -650.2728271, 621.2369385
3: -161.3594513, 583.3352661, -176.0654602, 639.4042969, -800.7636108, 759.4007568
4: -137.5561218, 533.8437500, -149.8268738, 585.1211548, -722.6772461, 683.6704712

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9756507
time: 1.16 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9756507
time: 1.14 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -129.1332550, 405.8225708, -139.5258179, 443.7532959, -572.8864746, 545.3483887
1: -180.8202820, 409.6738281, -195.7790985, 446.4605408, -627.2807617, 605.4529419
2: -152.7237396, 453.3137512, -165.4900818, 493.5649719, -646.2886963, 618.8038330
3: -161.2488556, 581.6445312, -174.5250549, 634.5444336, -795.7932739, 756.1695557
4: -137.3544464, 532.4979858, -148.7882080, 579.4626465, -716.8170166, 681.2861328

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9738482
time: 0.88 seconds

## Relational analysis of IS_A1_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9755471
time: 1.31 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -129.1332550, 405.8225708, -139.2723846, 442.7105713, -571.8438110, 545.0949707
1: -180.8202820, 409.6738281, -195.9212341, 445.7070923, -626.5273438, 605.5950928
2: -152.7237396, 453.3137512, -165.4566956, 492.7507629, -645.4744873, 618.7704468
3: -161.2488556, 581.6445312, -174.5605621, 633.3798218, -794.6286011, 756.2050781
4: -137.3544464, 532.4979858, -148.6779938, 578.6022949, -715.9566650, 681.1758423

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9738482
time: 1.00 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9755471
time: 1.00 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -135.6968231, 430.7904663, -133.1318359, 419.4592896, -555.1561279, 563.9223022
1: -190.1970978, 433.5797424, -186.1202240, 422.8794556, -613.0765381, 619.6999512
2: -160.7775116, 479.4364319, -157.3373566, 467.8631592, -628.6406250, 636.7738037
3: -169.6004028, 616.1744995, -166.0329590, 600.6323853, -770.2326660, 782.2073364
4: -144.5901337, 563.0349731, -141.5417938, 549.4505005, -694.0406494, 704.5767822

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9747454, upper bound: 554.9787050
time: 1.22 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9747454, upper bound: 554.9798758
time: 1.12 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -135.3625336, 429.4769592, -133.1318359, 419.4592896, -554.8218384, 562.6087646
1: -190.2272034, 432.5570984, -186.1202240, 422.8794556, -613.1066895, 618.6773071
2: -160.6483917, 478.3566895, -157.3373566, 467.8631592, -628.5115356, 635.6940308
3: -169.5371857, 614.6412354, -166.0329590, 600.6323853, -770.1695557, 780.6741333
4: -144.3938904, 561.8070068, -141.5417938, 549.4505005, -693.8443604, 703.3488159

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9747454, upper bound: 554.9787050
time: 1.23 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_B2

### Relational analysis result of IS_A1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9747454, upper bound: 554.9798758
time: 0.95 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -135.6968231, 430.7904663, -132.7345734, 417.8988953, -553.5957031, 563.5250244
1: -190.1970978, 433.5797424, -186.0755157, 421.6268921, -611.8239136, 619.6552734
2: -160.7775116, 479.4364319, -157.1767426, 466.5088501, -627.2863159, 636.6131592
3: -169.6004028, 616.1744995, -165.8930664, 598.7429199, -768.3432617, 782.0675659
4: -144.5901337, 563.0349731, -141.3210754, 547.9201050, -692.5102539, 704.3560791

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9747058, upper bound: 554.9787031
time: 1.34 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9747058, upper bound: 554.9797114
time: 0.99 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -135.3625336, 429.4769592, -132.7345734, 417.8988953, -553.2614136, 562.2115479
1: -190.2272034, 432.5570984, -186.0755157, 421.6268921, -611.8541260, 618.6326294
2: -160.6483917, 478.3566895, -157.1767426, 466.5088501, -627.1571655, 635.5334473
3: -169.5371857, 614.6412354, -165.8930664, 598.7429199, -768.2800903, 780.5343018
4: -144.3938904, 561.8070068, -141.3210754, 547.9201050, -692.3139648, 703.1280518

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9747058, upper bound: 554.9787031
time: 1.27 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9747058, upper bound: 554.9797114
time: 1.22 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -135.6968231, 430.7904663, -139.5258179, 443.7532959, -579.4500732, 570.3162842
1: -190.1970978, 433.5797424, -195.7790985, 446.4605408, -636.6575928, 629.3587646
2: -160.7775116, 479.4364319, -165.4900818, 493.5649719, -654.3424683, 644.9265137
3: -169.6004028, 616.1744995, -174.5250549, 634.5444336, -804.1448364, 790.6995239
4: -144.5901337, 563.0349731, -148.7882080, 579.4626465, -724.0527344, 711.8231812

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B2_A1_B1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9702253, upper bound: 554.9693018
time: 1.04 seconds

## Relational analysis of IS_A1_A2_B2_A1_B1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9702253, upper bound: 554.9716902
time: 1.13 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -135.6968231, 430.7904663, -139.2723846, 442.7105713, -578.4074097, 570.0628662
1: -190.1970978, 433.5797424, -195.9212341, 445.7070923, -635.9041748, 629.5009766
2: -160.7775116, 479.4364319, -165.4566956, 492.7507629, -653.5282593, 644.8931274
3: -169.6004028, 616.1744995, -174.5605621, 633.3798218, -802.9801025, 790.7350464
4: -144.5901337, 563.0349731, -148.6779938, 578.6022949, -723.1923218, 711.7128906

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B2_A1_B2_B1

### Relational analysis result of IS_A1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9702253, upper bound: 554.9693018
time: 1.09 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9702253, upper bound: 554.9716902
time: 1.31 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -135.3625336, 429.4769592, -139.5258179, 443.7532959, -579.1157837, 569.0027466
1: -190.2272034, 432.5570984, -195.7790985, 446.4605408, -636.6877441, 628.3361206
2: -160.6483917, 478.3566895, -165.4900818, 493.5649719, -654.2133789, 643.8467407
3: -169.5371857, 614.6412354, -174.5250549, 634.5444336, -804.0816040, 789.1662598
4: -144.3938904, 561.8070068, -148.7882080, 579.4626465, -723.8565063, 710.5951538

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684885, upper bound: 554.9688925
time: 1.26 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_B2

### Relational analysis result of IS_A1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684885, upper bound: 554.9709713
time: 1.12 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -135.3625336, 429.4769592, -139.2723846, 442.7105713, -578.0731201, 568.7493286
1: -190.2272034, 432.5570984, -195.9212341, 445.7070923, -635.9343262, 628.4783325
2: -160.6483917, 478.3566895, -165.4566956, 492.7507629, -653.3991699, 643.8133545
3: -169.5371857, 614.6412354, -174.5605621, 633.3798218, -802.9169312, 789.2017822
4: -144.3938904, 561.8070068, -148.6779938, 578.6022949, -722.9961548, 710.4848633

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684885, upper bound: 554.9688925
time: 1.02 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9684885, upper bound: 554.9709713
time: 1.32 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -133.0537109, 418.0853271, -131.0122681, 412.7502441, -545.8039551, 549.0975952
1: -185.2976227, 421.8851013, -183.5802307, 416.1888428, -601.4863281, 605.4652100
2: -156.6186371, 466.9302368, -155.0464935, 460.4937744, -617.1124268, 621.9766846
3: -165.4773560, 599.5919189, -163.6671600, 591.0204468, -756.4978027, 763.2589722
4: -140.9972992, 549.2918701, -139.3713989, 540.9598389, -681.9571533, 688.6632690

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9847058, upper bound: 554.9836222
time: 1.28 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9847058, upper bound: 554.9836222
time: 0.91 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -133.0537109, 418.0853271, -134.1653290, 421.9414673, -554.9950562, 552.2506714
1: -185.2976227, 421.8851013, -187.4551544, 425.6922913, -610.9897461, 609.3400879
2: -156.6186371, 466.9302368, -158.2877350, 471.1292114, -627.7478638, 625.2179565
3: -165.4773560, 599.5919189, -167.2518768, 604.9249878, -770.4022827, 766.8436890
4: -140.9972992, 549.2918701, -142.3607788, 554.4161987, -695.4134521, 691.6526489

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9847058, upper bound: 554.9842424
time: 1.15 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9847058, upper bound: 554.9842424
time: 1.25 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -132.3233948, 415.1637573, -128.9752045, 405.6633911, -537.9868164, 544.1389771
1: -184.7430267, 419.2759399, -180.0332184, 409.2827454, -594.0257568, 599.3091431
2: -156.0207367, 464.0734558, -152.1817017, 452.8391113, -608.8598633, 616.2551270
3: -164.8735657, 595.7244873, -160.6666718, 581.2777100, -746.1511841, 756.3911743
4: -140.3941803, 546.0523071, -136.9699707, 532.0159302, -672.4100952, 683.0221558

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9845512, upper bound: 554.9835734
time: 1.05 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9845512, upper bound: 554.9839292
time: 1.30 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -132.3233948, 415.1637573, -128.4959412, 403.7906494, -536.1140137, 543.6596680
1: -184.7430267, 419.2759399, -179.9012604, 407.6918030, -592.4347534, 599.1771851
2: -156.0207367, 464.0734558, -151.9429016, 451.1640015, -607.1847534, 616.0163574
3: -164.8735657, 595.7244873, -160.4413147, 578.9739990, -743.8474731, 756.1657715
4: -140.3941803, 546.0523071, -136.6783600, 530.1739502, -670.5681152, 682.7305298

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9845512, upper bound: 554.9835734
time: 1.51 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9845512, upper bound: 554.9839292
time: 1.18 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -133.0537109, 418.0853271, -137.2875824, 436.5627747, -569.6164551, 555.3729248
1: -185.2976227, 421.8851013, -193.0572510, 439.2462158, -624.5437622, 614.9423218
2: -156.6186371, 466.9302368, -163.0380554, 485.6901245, -642.3087769, 629.9682617
3: -165.4773560, 599.5919189, -172.0207977, 624.2588501, -789.7361450, 771.6126709
4: -140.9972992, 549.2918701, -146.4680939, 570.5666504, -711.5639648, 695.7599487

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9786191, upper bound: 554.9733778
time: 1.41 seconds

## Relational analysis of IS_A2_A1_B2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9786191, upper bound: 554.9733778
time: 1.34 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -133.0537109, 418.0853271, -140.8154144, 446.4977417, -579.5514526, 558.9007568
1: -185.2976227, 421.8851013, -197.4614410, 449.5711365, -634.8685913, 619.3464355
2: -156.6186371, 466.9302368, -166.6932220, 497.4051819, -654.0238037, 633.6234131
3: -165.4773560, 599.5919189, -176.0654602, 639.4042969, -804.8815918, 775.6573486
4: -140.9972992, 549.2918701, -149.8268738, 585.1211548, -726.1184692, 699.1186523

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9786191, upper bound: 554.9743703
time: 1.33 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9786191, upper bound: 554.9743703
time: 1.33 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -132.3233948, 415.1637573, -135.3168335, 429.9400024, -562.2634277, 550.4805908
1: -184.7430267, 419.2759399, -189.6072083, 432.7427368, -617.4857788, 608.8830566
2: -156.0207367, 464.0734558, -160.2491150, 478.4654236, -634.4861450, 624.3225708
3: -164.8735657, 595.7244873, -169.0794983, 615.1766968, -780.0501099, 764.8038940
4: -140.3941803, 546.0523071, -144.1440735, 561.8957520, -702.2899170, 690.1962891

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B2_A2_B1_B1

### Relational analysis result of IS_A2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9784923, upper bound: 554.9733209
time: 1.16 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_B2

### Relational analysis result of IS_A2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9784923, upper bound: 554.9740684
time: 0.98 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -132.3233948, 415.1637573, -134.9727783, 428.5489502, -560.8723145, 550.1365356
1: -184.7430267, 419.2759399, -189.6277466, 431.6169128, -616.3599243, 608.9036255
2: -156.0207367, 464.0734558, -160.1133270, 477.3164062, -633.3371582, 624.1867676
3: -164.8735657, 595.7244873, -169.0058441, 613.4851685, -778.3585815, 764.7303467
4: -140.3941803, 546.0523071, -143.9420471, 560.5874023, -700.9815674, 689.9942627

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B2_A2_B2_B1

### Relational analysis result of IS_A2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9784923, upper bound: 554.9733209
time: 1.04 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_B2

### Relational analysis result of IS_A2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9784923, upper bound: 554.9740684
time: 1.21 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -140.8572083, 446.5912781, -129.5068207, 407.2422180, -548.0993042, 576.0980835
1: -197.5076294, 449.6712646, -180.8375549, 410.8070984, -608.3146973, 630.5087891
2: -166.7368164, 497.5155334, -152.8676300, 454.5437622, -621.2805176, 650.3831787
3: -176.1089325, 639.5375977, -161.3594513, 583.3352661, -759.4442139, 800.8969727
4: -149.8696289, 585.2446289, -137.5561218, 533.8437500, -683.7132568, 722.8006592

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9756506, upper bound: 554.9772056
time: 1.25 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9756506, upper bound: 554.9772056
time: 1.23 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -140.8572083, 446.5912781, -133.0537109, 418.0853271, -558.9424438, 579.6450195
1: -197.5076294, 449.6712646, -185.2976227, 421.8851013, -619.3926392, 634.9688721
2: -166.7368164, 497.5155334, -156.6186371, 466.9302368, -633.6669922, 654.1341553
3: -176.1089325, 639.5375977, -165.4773560, 599.5919189, -775.7008057, 805.0149536
4: -149.8696289, 585.2446289, -140.9972992, 549.2918701, -699.1614380, 726.2418823

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9756507, upper bound: 554.9784350
time: 1.31 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9756507, upper bound: 554.9784350
time: 1.17 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -139.2046967, 440.9418640, -128.4959412, 403.7906494, -542.9952393, 569.4377441
1: -194.6376648, 444.0958252, -179.9012604, 407.6918030, -602.3292847, 623.9970703
2: -164.4753876, 491.3377380, -151.9429016, 451.1640015, -615.6393433, 643.2805176
3: -173.6896820, 631.5960083, -160.4413147, 578.9739990, -752.6636353, 792.0373535
4: -147.9734955, 577.8323364, -136.6783600, 530.1739502, -678.1474609, 714.5106201

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9755471, upper bound: 554.9772056
time: 1.37 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9755471, upper bound: 554.9782859
time: 1.28 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -138.9383545, 439.6115417, -128.4959412, 403.7906494, -542.7290039, 568.1074829
1: -194.6855316, 443.0742798, -179.9012604, 407.6918030, -602.3773193, 622.9755249
2: -164.3618774, 490.2503662, -151.9429016, 451.1640015, -615.5258179, 642.1932373
3: -173.6382446, 630.0073853, -160.4413147, 578.9739990, -752.6122437, 790.4487305
4: -147.8123169, 576.6052856, -136.6783600, 530.1739502, -677.9862061, 713.2835693

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9755471, upper bound: 554.9772056
time: 1.32 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9755471, upper bound: 554.9782859
time: 1.26 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -139.2046967, 440.9418640, -137.2875824, 436.5627747, -575.7673950, 578.2293701
1: -194.6376648, 444.0958252, -193.0572510, 439.2462158, -633.8838501, 637.1530151
2: -164.4753876, 491.3377380, -163.0380554, 485.6901245, -650.1654053, 654.3757324
3: -173.6896820, 631.5960083, -172.0207977, 624.2588501, -797.9484253, 803.6167603
4: -147.9734955, 577.8323364, -146.4680939, 570.5666504, -718.5401611, 724.3004150

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708176, upper bound: 554.9678801
time: 1.18 seconds

## Relational analysis of IS_A2_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708176, upper bound: 554.9678801
time: 1.46 seconds

## BFS IS instance: IS_A2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -139.2046967, 440.9418640, -140.8154144, 446.4977417, -585.7024536, 581.7572632
1: -194.6376648, 444.0958252, -197.4614410, 449.5711365, -644.2086792, 641.5571899
2: -164.4753876, 491.3377380, -166.6932220, 497.4051819, -661.8804932, 658.0308228
3: -173.6896820, 631.5960083, -176.0654602, 639.4042969, -813.0938721, 807.6614990
4: -147.9734955, 577.8323364, -149.8268738, 585.1211548, -733.0946655, 727.6591187

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708176, upper bound: 554.9692813
time: 1.19 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708176, upper bound: 554.9692813
time: 1.11 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -138.9383545, 439.6115417, -135.3168335, 429.9400024, -568.8783569, 574.9283447
1: -194.6855316, 443.0742798, -189.6072083, 432.7427368, -627.4282837, 632.6813965
2: -164.3618774, 490.2503662, -160.2491150, 478.4654236, -642.8272705, 650.4995117
3: -173.6382446, 630.0073853, -169.0794983, 615.1766968, -788.8148804, 799.0867920
4: -147.8123169, 576.6052856, -144.1440735, 561.8957520, -709.7079468, 720.7493286

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699661, upper bound: 554.9674190
time: 1.24 seconds

## Relational analysis of IS_A2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699662, upper bound: 554.9689722
time: 1.16 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -138.9383545, 439.6115417, -134.9727783, 428.5489502, -567.4873047, 574.5843506
1: -194.6855316, 443.0742798, -189.6277466, 431.6169128, -626.3024292, 632.7019653
2: -164.3618774, 490.2503662, -160.1133270, 477.3164062, -641.6781616, 650.3637085
3: -173.6382446, 630.0073853, -169.0058441, 613.4851685, -787.1233521, 799.0132446
4: -147.8123169, 576.6052856, -143.9420471, 560.5874023, -708.3996582, 720.5473022

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699661, upper bound: 554.9674190
time: 1.29 seconds

## Relational analysis of IS_A2_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699662, upper bound: 554.9689722
time: 1.26 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.18 seconds
IS_A1_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9849517
IS_A1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9857507
IS_A1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9849517
IS_A1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9857507
IS_A1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9849565
IS_A1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9857245
IS_A1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9849565
IS_A1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9835734, upper bound: 554.9857245
IS_A1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9739138
IS_A1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9739138
IS_A1_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9756507
IS_A1_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9756507
IS_A1_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9738482
IS_A1_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9755471
IS_A1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9738482
IS_A1_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9772056, upper bound: 554.9755471
IS_A1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9747454, upper bound: 554.9787050
IS_A1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9747454, upper bound: 554.9798758
IS_A1_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9747454, upper bound: 554.9787050
IS_A1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9747454, upper bound: 554.9798758
IS_A1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9747058, upper bound: 554.9787031
IS_A1_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9747058, upper bound: 554.9797114
IS_A1_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9747058, upper bound: 554.9787031
IS_A1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9747058, upper bound: 554.9797114
IS_A1_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9702253, upper bound: 554.9693018
IS_A1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9702253, upper bound: 554.9716902
IS_A1_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9702253, upper bound: 554.9693018
IS_A1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9702253, upper bound: 554.9716902
IS_A1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9684885, upper bound: 554.9688925
IS_A1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9684885, upper bound: 554.9709713
IS_A1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9684885, upper bound: 554.9688925
IS_A1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9684885, upper bound: 554.9709713
IS_A2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9847058, upper bound: 554.9836222
IS_A2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9847058, upper bound: 554.9836222
IS_A2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9847058, upper bound: 554.9842424
IS_A2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9847058, upper bound: 554.9842424
IS_A2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9845512, upper bound: 554.9835734
IS_A2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9845512, upper bound: 554.9839292
IS_A2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9845512, upper bound: 554.9835734
IS_A2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9845512, upper bound: 554.9839292
IS_A2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9786191, upper bound: 554.9733778
IS_A2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9786191, upper bound: 554.9733778
IS_A2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9786191, upper bound: 554.9743703
IS_A2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9786191, upper bound: 554.9743703
IS_A2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9784923, upper bound: 554.9733209
IS_A2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9784923, upper bound: 554.9740684
IS_A2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9784923, upper bound: 554.9733209
IS_A2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9784923, upper bound: 554.9740684
IS_A2_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9756506, upper bound: 554.9772056
IS_A2_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9756506, upper bound: 554.9772056
IS_A2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9756507, upper bound: 554.9784350
IS_A2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9756507, upper bound: 554.9784350
IS_A2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9755471, upper bound: 554.9772056
IS_A2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9755471, upper bound: 554.9782859
IS_A2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9755471, upper bound: 554.9772056
IS_A2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9755471, upper bound: 554.9782859
IS_A2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9708176, upper bound: 554.9678801
IS_A2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9708176, upper bound: 554.9678801
IS_A2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9708176, upper bound: 554.9692813
IS_A2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9708176, upper bound: 554.9692813
IS_A2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9699661, upper bound: 554.9674190
IS_A2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9699662, upper bound: 554.9689722
IS_A2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9699661, upper bound: 554.9674190
IS_A2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -554.9699662, upper bound: 554.9689722

## BFS IS instance: IS_A1_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -129.5068207, 407.2422180, -129.5068207, 407.2422180, -536.7490234, 536.7490234
1: -180.8375549, 410.8070984, -180.8375549, 410.8070984, -591.6445312, 591.6445312
2: -152.8676300, 454.5437622, -152.8676300, 454.5437622, -607.4113770, 607.4113770
3: -161.3594513, 583.3352661, -161.3594513, 583.3352661, -744.6946411, 744.6945801
4: -137.5561218, 533.8437500, -137.5561218, 533.8437500, -671.3997803, 671.3997803

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9834754, upper bound: 554.9849397
time: 1.16 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9834696, upper bound: 554.9834696
time: 1.26 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -129.5068207, 407.2422180, -133.0537109, 418.0853271, -547.5921631, 540.2958984
1: -180.8375549, 410.8070984, -185.2976227, 421.8851013, -602.7224731, 596.1046143
2: -152.8676300, 454.5437622, -156.6186371, 466.9302368, -619.7978516, 611.1624146
3: -161.3594513, 583.3352661, -165.4773560, 599.5919189, -760.9512329, 748.8126221
4: -137.5561218, 533.8437500, -140.9972992, 549.2918701, -686.8479614, 674.8410034

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9023589, upper bound: 554.9826481
time: 1.23 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.7637397, upper bound: 554.9047825
time: 0.99 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -129.5068207, 407.2422180, -129.1332550, 405.8225708, -535.3294067, 536.3754272
1: -180.8375549, 410.8070984, -180.8202820, 409.6738281, -590.5113525, 591.6272583
2: -152.8676300, 454.5437622, -152.7237396, 453.3137512, -606.1813965, 607.2675171
3: -161.3594513, 583.3352661, -161.2488556, 581.6445312, -743.0038452, 744.5841064
4: -137.5561218, 533.8437500, -137.3544464, 532.4979858, -670.0540771, 671.1981201

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835719, upper bound: 554.9849497
time: 1.46 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835639, upper bound: 554.9834745
time: 1.00 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -129.5068207, 407.2422180, -132.3233948, 415.1637573, -544.6705933, 539.5656128
1: -180.8375549, 410.8070984, -184.7430267, 419.2759399, -600.1134644, 595.5501099
2: -152.8676300, 454.5437622, -156.0207367, 464.0734558, -616.9411011, 610.5645142
3: -161.3594513, 583.3352661, -164.8735657, 595.7244873, -757.0838013, 748.2086792
4: -137.5561218, 533.8437500, -140.3941803, 546.0523071, -683.6082764, 674.2378540

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9002698, upper bound: 554.9823023
time: 1.22 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.7630860, upper bound: 554.9038662
time: 1.01 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -129.1332550, 405.8225708, -129.5068207, 407.2422180, -536.3754272, 535.3293457
1: -180.8202820, 409.6738281, -180.8375549, 410.8070984, -591.6273193, 590.5113525
2: -152.7237396, 453.3137512, -152.8676300, 454.5437622, -607.2675171, 606.1813965
3: -161.2488556, 581.6445312, -161.3594513, 583.3352661, -744.5841064, 743.0038452
4: -137.3544464, 532.4979858, -137.5561218, 533.8437500, -671.1981201, 670.0540771

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_A2_B1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9836211, upper bound: 554.9843156
time: 1.18 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9834745, upper bound: 554.9843063
time: 1.00 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -129.1332550, 405.8225708, -133.0537109, 418.0853271, -547.2185669, 538.8762207
1: -180.8202820, 409.6738281, -185.2976227, 421.8851013, -602.7052612, 594.9714355
2: -152.7237396, 453.3137512, -156.6186371, 466.9302368, -619.6539917, 609.9323730
3: -161.2488556, 581.6445312, -165.4773560, 599.5919189, -760.8407593, 747.1218262
4: -137.3544464, 532.4979858, -140.9972992, 549.2918701, -686.6463013, 673.4953003

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9021404, upper bound: 554.9825694
time: 1.02 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.7609785, upper bound: 554.9027400
time: 1.15 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -129.1332550, 405.8225708, -129.1332550, 405.8225708, -534.9557495, 534.9557495
1: -180.8202820, 409.6738281, -180.8202820, 409.6738281, -590.4941406, 590.4941406
2: -152.7237396, 453.3137512, -152.7237396, 453.3137512, -606.0374756, 606.0374756
3: -161.2488556, 581.6445312, -161.2488556, 581.6445312, -742.8933716, 742.8933716
4: -137.3544464, 532.4979858, -137.3544464, 532.4979858, -669.8524170, 669.8524170

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835717, upper bound: 554.9843156
time: 1.25 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9835479, upper bound: 554.9843155
time: 1.68 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -129.1332550, 405.8225708, -132.3233948, 415.1637573, -544.2969971, 538.1459961
1: -180.8202820, 409.6738281, -184.7430267, 419.2759399, -600.0961914, 594.4168701
2: -152.7237396, 453.3137512, -156.0207367, 464.0734558, -616.7971802, 609.3344727
3: -161.2488556, 581.6445312, -164.8735657, 595.7244873, -756.9733276, 746.5179443
4: -137.3544464, 532.4979858, -140.3941803, 546.0523071, -683.4066162, 672.8921509

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_A2_B2_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9000595, upper bound: 554.9822171
time: 1.46 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.7603248, upper bound: 554.9018811
time: 1.18 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -129.5068207, 407.2422180, -135.6968231, 430.7904663, -560.2973022, 542.9389648
1: -180.8375549, 410.8070984, -190.1970978, 433.5797424, -614.4172363, 601.0040894
2: -152.8676300, 454.5437622, -160.7775116, 479.4364319, -632.3040771, 615.3212891
3: -161.3594513, 583.3352661, -169.6004028, 616.1744995, -777.5338135, 752.9355469
4: -137.5561218, 533.8437500, -144.5901337, 563.0349731, -700.5910645, 678.4337158

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.8483660, upper bound: 554.7306337
time: 1.24 seconds

## Relational analysis of IS_A1_A1_B2_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_A1_B1_B1_B1

### Relational analysis result of IS_A1_A1_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9784869, upper bound: 554.9735561
time: 2.41 seconds

## Relational analysis of IS_A1_A1_B2_A1_B1_B1_B2

### Relational analysis result of IS_A1_A1_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9787098, upper bound: 554.9746114
time: 1.21 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -129.5068207, 407.2422180, -135.3625336, 429.4769592, -558.9837646, 542.6046753
1: -180.8375549, 410.8070984, -190.2272034, 432.5570984, -613.3945923, 601.0343018
2: -152.8676300, 454.5437622, -160.6483917, 478.3566895, -631.2243042, 615.1921387
3: -161.3594513, 583.3352661, -169.5371857, 614.6412354, -776.0006104, 752.8724365
4: -137.5561218, 533.8437500, -144.3938904, 561.8070068, -699.3630371, 678.2376099

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.8483660, upper bound: 554.7306337
time: 1.37 seconds

## Relational analysis of IS_A1_A1_B2_A1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9784869, upper bound: 554.9735561
time: 0.94 seconds

## Relational analysis of IS_A1_A1_B2_A1_B1_B2_B2

### Relational analysis result of IS_A1_A1_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9787098, upper bound: 554.9746114
time: 1.54 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -129.5068207, 407.2422180, -139.1941528, 440.9164734, -570.4232788, 546.4363403
1: -180.8375549, 410.8070984, -194.6246796, 444.0685730, -624.9060669, 605.4317017
2: -152.8676300, 454.5437622, -164.4633179, 491.3078003, -644.1754150, 619.0070801
3: -161.3594513, 583.3352661, -173.6775665, 631.5599365, -792.9192505, 757.0128174
4: -137.5561218, 533.8437500, -147.9617157, 577.7984619, -715.3544922, 681.8054199

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9745732, upper bound: 554.9754091
time: 1.00 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.7893744, upper bound: 554.8944554
time: 1.12 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -129.5068207, 407.2422180, -138.9383545, 439.6115417, -569.1183472, 546.1805420
1: -180.8375549, 410.8070984, -194.6855316, 443.0742798, -623.9117432, 605.4926147
2: -152.8676300, 454.5437622, -164.3618774, 490.2503662, -643.1179810, 618.9056396
3: -161.3594513, 583.3352661, -173.6382446, 630.0073853, -791.3666992, 756.9735107
4: -137.5561218, 533.8437500, -147.8123169, 576.6052856, -714.1613770, 681.6559448

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9745732, upper bound: 554.9754091
time: 1.45 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.7893744, upper bound: 554.8944554
time: 1.53 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -129.1332550, 405.8225708, -135.6968231, 430.7904663, -559.9237061, 541.5193481
1: -180.8202820, 409.6738281, -190.1970978, 433.5797424, -614.3999634, 599.8709106
2: -152.7237396, 453.3137512, -160.7775116, 479.4364319, -632.1601562, 614.0912476
3: -161.2488556, 581.6445312, -169.6004028, 616.1744995, -777.4233398, 751.2448730
4: -137.3544464, 532.4979858, -144.5901337, 563.0349731, -700.3894043, 677.0880127

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9777056, upper bound: 554.9757941
time: 0.99 seconds

## Relational analysis of IS_A1_A1_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9777020, upper bound: 554.9756640
time: 1.19 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -129.1332550, 405.8225708, -139.1941528, 440.9164734, -570.0496216, 545.0167236
1: -180.8202820, 409.6738281, -194.6246796, 444.0685730, -624.8887939, 604.2985229
2: -152.7237396, 453.3137512, -164.4633179, 491.3078003, -644.0315552, 617.7770996
3: -161.2488556, 581.6445312, -173.6775665, 631.5599365, -792.8087769, 755.3220825
4: -137.3544464, 532.4979858, -147.9617157, 577.7984619, -715.1528320, 680.4596558

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9290807, upper bound: 554.9761441
time: 1.28 seconds

## Relational analysis of IS_A1_A1_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.7624684, upper bound: 554.7861718
time: 1.05 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -129.1332550, 405.8225708, -135.3625336, 429.4769592, -558.6102295, 541.1850586
1: -180.8202820, 409.6738281, -190.2272034, 432.5570984, -613.3773193, 599.9010010
2: -152.7237396, 453.3137512, -160.6483917, 478.3566895, -631.0804443, 613.9621582
3: -161.2488556, 581.6445312, -169.5371857, 614.6412354, -775.8900757, 751.1817017
4: -137.3544464, 532.4979858, -144.3938904, 561.8070068, -699.1613770, 676.8918457

Time for backsubstitution: 1.99 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=619.6494140625
rel_dist={0: [-554.9907236424904, 554.9907236424906]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.7335056, upper bound: 554.7990778
time: 1.07 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.7145819, upper bound: 554.7145819
time: 1.28 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.53 seconds
IS_B1, status: Status.VERIFIED, split count: 1, time: 2.53
Output dim: 0, lower bound: -554.7335056, upper bound: 554.7990778
IS_B2, status: Status.VERIFIED, split count: 1, time: 2.53
Output dim: 0, lower bound: -554.7145819, upper bound: 554.7145819
Binary search (step 2): status=Status.VERIFIED, low=0.1250000, high=0.2500000, mid=0.1250000, abs_max=619.6494140625
rel_dist={0: [-554.990153595014, 554.9901535950139]}

## Binary search (step 3) starts
Candidate diff: 0.1875000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.7422943, upper bound: 554.8335855
time: 0.95 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.7145992, upper bound: 554.7145992
time: 1.00 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.13 seconds
IS_B1, status: Status.VERIFIED, split count: 1, time: 2.13
Output dim: 0, lower bound: -554.7422943, upper bound: 554.8335855
IS_B2, status: Status.VERIFIED, split count: 1, time: 2.13
Output dim: 0, lower bound: -554.7145992, upper bound: 554.7145992
Binary search (step 3): status=Status.VERIFIED, low=0.1875000, high=0.2500000, mid=0.1875000, abs_max=619.6494140625
rel_dist={0: [-554.9905301746055, 554.9905301746055]}

## Binary search (step 4) starts
Candidate diff: 0.2187500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9840393, upper bound: 554.9861635
time: 0.97 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9853031, upper bound: 554.9853031
time: 0.99 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.13 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.13
Output dim: 0, lower bound: -554.9840393, upper bound: 554.9861635
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.13
Output dim: 0, lower bound: -554.9853031, upper bound: 554.9853031

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -144.4209442, 458.6982422, -148.1843414, 471.4650574, -615.8859863, 606.8823853
1: -202.9677582, 461.6006775, -208.4494476, 474.2722168, -677.2398071, 670.0501099
2: -171.4153137, 510.5334778, -176.0516663, 524.4273071, -695.8425293, 686.5851440
3: -180.9070740, 655.8257446, -185.7463837, 673.9080200, -854.8150024, 841.5721436
4: -154.0161438, 599.6799927, -158.1413727, 615.8510742, -769.8671875, 757.8212891

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9840393, upper bound: 554.9840393
time: 1.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9840393, upper bound: 554.9853031
time: 1.35 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -147.6497955, 467.7235718, -143.3740234, 455.6749878, -603.3246460, 611.0974121
1: -206.9226227, 471.0274048, -201.4233398, 458.5483398, -665.4709473, 672.4507446
2: -174.7036133, 521.2171631, -170.0857086, 507.1589661, -681.8624878, 691.3028564
3: -184.5714264, 669.6788330, -179.5448151, 651.7089844, -836.2803955, 849.2236328
4: -157.0737762, 613.0461426, -152.8522644, 595.7423706, -752.8161621, 765.8983154

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9791933, upper bound: 554.9777382
time: 1.11 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723811, upper bound: 554.9723811
time: 1.22 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.52 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.52
Output dim: 0, lower bound: -554.9840393, upper bound: 554.9840393
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.52
Output dim: 0, lower bound: -554.9840393, upper bound: 554.9853031
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 4.52
Output dim: 0, lower bound: -554.9791933, upper bound: 554.9777382
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 4.52
Output dim: 0, lower bound: -554.9723811, upper bound: 554.9723811

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -144.4209442, 458.6982422, -144.4209442, 458.6982422, -603.1191406, 603.1191406
1: -202.9677582, 461.6006775, -202.9677582, 461.6006775, -664.5682983, 664.5682983
2: -171.4153137, 510.5334778, -171.4153137, 510.5334778, -681.9487305, 681.9487305
3: -180.9070740, 655.8257446, -180.9070740, 655.8257446, -836.7327881, 836.7327881
4: -154.0161438, 599.6799927, -154.0161438, 599.6799927, -753.6960449, 753.6960449

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9838483, upper bound: 554.9850475
time: 1.03 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9837081, upper bound: 554.9850475
time: 1.26 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -144.4209442, 458.6982422, -147.6497955, 467.7235718, -612.1445312, 606.3478394
1: -202.9677582, 461.6006775, -206.9226227, 471.0274048, -673.9951782, 668.5231323
2: -171.4153137, 510.5334778, -174.7036133, 521.2171631, -692.6324463, 685.2370605
3: -180.9070740, 655.8257446, -184.5714264, 669.6788330, -850.5858765, 840.3971558
4: -154.0161438, 599.6799927, -157.0737762, 613.0461426, -767.0621338, 756.7537842

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9767049, upper bound: 554.9804378
time: 1.13 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9709673, upper bound: 554.9736288
time: 1.20 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -134.9726562, 424.7419739, -141.9582214, 450.9740906, -585.9467773, 566.7001953
1: -188.6135559, 428.4528198, -199.4303131, 453.8532410, -642.4667969, 627.8831177
2: -159.2701263, 474.2145996, -168.4089203, 501.9511108, -661.2210693, 642.6235352
3: -168.2837219, 608.8666992, -177.7557373, 644.9962769, -813.2800293, 786.6224365
4: -143.2449951, 557.9903564, -151.3425751, 589.5944824, -732.8394165, 709.3328857

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723811, upper bound: 554.9723811
time: 1.31 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723811, upper bound: 554.9723811
time: 1.10 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -141.6680908, 449.4027405, -141.0438843, 448.5088501, -590.1768799, 590.4466553
1: -198.6617737, 452.4513550, -198.1963654, 451.3100891, -649.9718628, 650.6477051
2: -167.7134552, 500.6373596, -167.3516541, 499.1317749, -666.8452148, 667.9888306
3: -177.1431580, 643.5125732, -176.6433105, 641.4844971, -818.6275635, 820.1558838
4: -150.7537537, 588.8465576, -150.3897705, 586.3080444, -737.0617676, 739.2362061

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723811, upper bound: 554.9723811
time: 1.01 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723811, upper bound: 554.9723811
time: 1.07 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.29 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 0, lower bound: -554.9838483, upper bound: 554.9850475
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 0, lower bound: -554.9837081, upper bound: 554.9850475
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 0, lower bound: -554.9767049, upper bound: 554.9804378
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 0, lower bound: -554.9709673, upper bound: 554.9736288
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 0, lower bound: -554.9723811, upper bound: 554.9723811
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 0, lower bound: -554.9723811, upper bound: 554.9723811
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 0, lower bound: -554.9723811, upper bound: 554.9723811
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 0, lower bound: -554.9723811, upper bound: 554.9723811

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -143.5416260, 455.6802979, -142.2265778, 450.8411255, -594.3827515, 597.9068604
1: -201.7112122, 458.5924683, -199.2681580, 453.8541870, -655.5654297, 657.8605957
2: -170.3549347, 507.1656799, -168.4452820, 501.9418640, -672.2968140, 675.6109619
3: -179.7837982, 651.5463257, -177.7439117, 644.7758179, -824.5596313, 829.2902222
4: -153.0616150, 595.7919312, -151.4957886, 589.4702759, -742.5317993, 747.2877197

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.8464767, upper bound: 554.7447960
time: 0.97 seconds

## Relational analysis of IS_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9788454, upper bound: 554.9773035
time: 1.35 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9706410, upper bound: 554.9718688
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -144.4209442, 458.6982422, -141.7474823, 449.0274963, -593.4484253, 600.4455566
1: -202.9677582, 461.6006775, -199.0686035, 452.3401794, -655.3078003, 660.6691895
2: -171.4153137, 510.5334778, -168.1255951, 500.3099365, -671.7251587, 678.6590576
3: -180.9070740, 655.8257446, -177.4720612, 642.5354614, -823.4425049, 833.2977905
4: -154.0161438, 599.6799927, -151.1334991, 587.5854492, -741.6014404, 750.8134766

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9850475, upper bound: 554.9850331
time: 1.05 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9850475, upper bound: 554.9850475
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -143.0493469, 454.1390991, -134.9726562, 424.7419739, -567.7913208, 589.1117554
1: -201.0359497, 457.0296326, -188.6135559, 428.4528198, -629.4887695, 645.6431885
2: -169.7879791, 505.4473267, -159.2701263, 474.2145996, -644.0025635, 664.7173462
3: -179.1695251, 649.2662964, -168.2837219, 608.8666992, -788.0361328, 817.5500488
4: -152.5530853, 593.6414795, -143.2449951, 557.9903564, -710.5434570, 736.8864136

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9709673, upper bound: 554.9736288
time: 1.23 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9709673, upper bound: 554.9736288
time: 1.29 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -142.0776825, 451.5132751, -141.6680908, 449.4027405, -591.4804077, 593.1813354
1: -199.7247162, 454.3278809, -198.6617737, 452.4513550, -652.1760254, 652.9896240
2: -168.6699524, 502.4600220, -167.7134552, 500.6373596, -669.3071899, 670.1734619
3: -177.9928894, 645.5706787, -177.1431580, 643.5125732, -821.5054932, 822.7138672
4: -151.5412292, 590.2041016, -150.7537537, 588.8465576, -740.3876953, 740.9578247

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9709673, upper bound: 554.9736288
time: 1.33 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9709673, upper bound: 554.9736288
time: 0.90 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -134.9726562, 424.7419739, -130.6123657, 411.6246338, -546.5972900, 555.3543091
1: -188.6135559, 428.4528198, -182.9770508, 415.0874329, -603.7008667, 611.4298706
2: -159.2701263, 474.2145996, -154.5317688, 459.3524170, -618.6224365, 628.7463379
3: -168.2837219, 608.8666992, -163.1557159, 589.5997925, -757.8835449, 772.0223999
4: -143.2449951, 557.9903564, -138.9412231, 539.7780151, -683.0229492, 696.9315186

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9791933, upper bound: 554.9762731
time: 1.44 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9791933, upper bound: 554.9771026
time: 1.52 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -134.9726562, 424.7419739, -137.0725098, 436.2905273, -571.2631226, 561.8144531
1: -188.6135559, 428.4528198, -192.6868896, 438.9768982, -627.5904541, 621.1397095
2: -159.2701263, 474.2145996, -162.6926117, 485.4533997, -644.7233887, 636.9072266
3: -168.2837219, 608.8666992, -171.7039490, 624.0643921, -792.3481445, 780.5706787
4: -143.2449951, 557.9903564, -146.1962280, 570.2287598, -713.4736938, 704.1865234

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9791933, upper bound: 554.9762731
time: 1.22 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9791933, upper bound: 554.9771026
time: 1.26 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -141.6680908, 449.4027405, -130.6123657, 411.6246338, -553.2927246, 580.0150757
1: -198.6617737, 452.4513550, -182.9770508, 415.0874329, -613.7492065, 635.4284058
2: -167.7134552, 500.6373596, -154.5317688, 459.3524170, -627.0657959, 655.1690674
3: -177.1431580, 643.5125732, -163.1557159, 589.5997925, -766.7428589, 806.6682739
4: -150.7537537, 588.8465576, -138.9412231, 539.7780151, -690.5316772, 727.7877808

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9704303, upper bound: 554.9708104
time: 1.03 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699589, upper bound: 554.9699590
time: 1.18 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -141.6680908, 449.4027405, -137.0725098, 436.2905273, -577.9584961, 586.4752197
1: -198.6617737, 452.4513550, -192.6868896, 438.9768982, -637.6386719, 645.1382446
2: -167.7134552, 500.6373596, -162.6926117, 485.4533997, -653.1668091, 663.3298950
3: -177.1431580, 643.5125732, -171.7039490, 624.0643921, -801.2075195, 815.2165527
4: -150.7537537, 588.8465576, -146.1962280, 570.2287598, -720.9824829, 735.0427856

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708104, upper bound: 554.9704303
time: 1.15 seconds

## Relational analysis of IS_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699589, upper bound: 554.9699590
time: 1.21 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.68 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 0, lower bound: -554.9788454, upper bound: 554.9773035
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 0, lower bound: -554.9706410, upper bound: 554.9718688
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 0, lower bound: -554.9850475, upper bound: 554.9850331
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 0, lower bound: -554.9850475, upper bound: 554.9850475
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 0, lower bound: -554.9709673, upper bound: 554.9736288
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 0, lower bound: -554.9709673, upper bound: 554.9736288
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 0, lower bound: -554.9709673, upper bound: 554.9736288
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 0, lower bound: -554.9709673, upper bound: 554.9736288
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 0, lower bound: -554.9791933, upper bound: 554.9762731
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 0, lower bound: -554.9791933, upper bound: 554.9771026
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 0, lower bound: -554.9791933, upper bound: 554.9762731
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 0, lower bound: -554.9791933, upper bound: 554.9771026
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 0, lower bound: -554.9704303, upper bound: 554.9708104
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 0, lower bound: -554.9699589, upper bound: 554.9699590
IS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 0, lower bound: -554.9708104, upper bound: 554.9704303
IS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.68
Output dim: 0, lower bound: -554.9699589, upper bound: 554.9699590

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -130.8796387, 412.2967529, -140.8496704, 446.2448425, -577.1245117, 553.1464233
1: -183.3904114, 415.7405396, -197.3220520, 449.2522888, -632.6427002, 613.0625610
2: -154.8859406, 459.9907227, -166.8048248, 496.8097229, -651.6956787, 626.7954102
3: -163.4978790, 590.3821411, -175.9925079, 638.1691284, -801.6669922, 766.3746338
4: -139.2274323, 540.3772583, -150.0195465, 583.3884277, -722.6158447, 690.3967896

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9706410, upper bound: 554.9718688
time: 1.16 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9706410, upper bound: 554.9718688
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -137.1628418, 436.1359253, -139.8300323, 443.4956665, -580.6585083, 575.9657593
1: -192.8791199, 438.8210144, -195.9462128, 446.4235229, -639.3026123, 634.7672119
2: -162.8877106, 485.2140198, -165.6338043, 493.6914673, -656.5791626, 650.8477783
3: -171.8616943, 623.6537476, -174.7569427, 634.2959595, -806.1576538, 798.4107056
4: -146.3327789, 570.0169067, -148.9625854, 579.7820435, -726.1148071, 718.9794922

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9706410, upper bound: 554.9718688
time: 1.19 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9706410, upper bound: 554.9718688
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -142.2265778, 450.8411255, -141.7474823, 449.0274963, -591.2540283, 592.5886230
1: -199.2681580, 453.8541870, -199.0686035, 452.3401794, -651.6082153, 652.9227905
2: -168.4452820, 501.9418640, -168.1255951, 500.3099365, -668.7552490, 670.0674438
3: -177.7439117, 644.7758179, -177.4720612, 642.5354614, -820.2793579, 822.2478638
4: -151.4957886, 589.4702759, -151.1334991, 587.5854492, -739.0811768, 740.6037598

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9756812, upper bound: 554.9781013
time: 0.87 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9702367, upper bound: 554.9702367
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -141.7474823, 449.0274963, -141.7474823, 449.0274963, -590.7749023, 590.7749023
1: -199.0686035, 452.3401794, -199.0686035, 452.3401794, -651.4086914, 651.4086914
2: -168.1255951, 500.3099365, -168.1255951, 500.3099365, -668.4355469, 668.4355469
3: -177.4720612, 642.5354614, -177.4720612, 642.5354614, -820.0075073, 820.0075073
4: -151.1334991, 587.5854492, -151.1334991, 587.5854492, -738.7189331, 738.7188721

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9781013, upper bound: 554.9756881
time: 1.15 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9702367, upper bound: 554.9702367
time: 1.67 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -131.8264923, 415.5336609, -134.9726562, 424.7419739, -556.5684814, 550.5063477
1: -184.7457886, 418.9423218, -188.6135559, 428.4528198, -613.1986084, 607.5557861
2: -156.0319977, 463.5839844, -159.2701263, 474.2145996, -630.2465820, 622.8539429
3: -164.7066956, 594.9381104, -168.2837219, 608.8666992, -773.5733032, 763.2218018
4: -140.2548218, 544.5361328, -143.2449951, 557.9903564, -698.2451172, 687.7810669

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9746680, upper bound: 554.9794736
time: 1.40 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9746340, upper bound: 554.9792793
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -138.0580139, 439.1935120, -134.9726562, 424.7419739, -562.7999878, 574.1661377
1: -194.1572723, 441.8686218, -188.6135559, 428.4528198, -622.6100464, 630.4821167
2: -163.9659271, 488.6260071, -159.2701263, 474.2145996, -638.1805420, 647.8959351
3: -173.0027618, 627.9880371, -168.2837219, 608.8666992, -781.8693848, 796.2717285
4: -147.3025970, 573.9549561, -143.2449951, 557.9903564, -705.2929688, 717.1998901

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9746680, upper bound: 554.9794736
time: 0.92 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9746340, upper bound: 554.9792793
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -131.8264923, 415.5336609, -141.6680908, 449.4027405, -581.2292480, 557.2017822
1: -184.7457886, 418.9423218, -198.6617737, 452.4513550, -637.1971436, 617.6041260
2: -156.0319977, 463.5839844, -167.7134552, 500.6373596, -656.6691284, 631.2973633
3: -164.7066956, 594.9381104, -177.1431580, 643.5125732, -808.2192383, 772.0811157
4: -140.2548218, 544.5361328, -150.7537537, 588.8465576, -729.1012573, 695.2897949

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9700542, upper bound: 554.9714536
time: 1.02 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683134, upper bound: 554.9707864
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -138.0580139, 439.1935120, -141.6680908, 449.4027405, -587.4607544, 580.8615723
1: -194.1572723, 441.8686218, -198.6617737, 452.4513550, -646.6085815, 640.5303955
2: -163.9659271, 488.6260071, -167.7134552, 500.6373596, -664.6032104, 656.3392944
3: -173.0027618, 627.9880371, -177.1431580, 643.5125732, -816.5153198, 805.1311035
4: -147.3025970, 573.9549561, -150.7537537, 588.8465576, -736.1491089, 724.7086792

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9688047, upper bound: 554.9716982
time: 1.21 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683134, upper bound: 554.9707864
time: 1.30 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -134.9726562, 424.7419739, -131.8264923, 415.5336609, -550.5063477, 556.5684814
1: -188.6135559, 428.4528198, -184.7457886, 418.9423218, -607.5557251, 613.1986084
2: -159.2701263, 474.2145996, -156.0319977, 463.5839844, -622.8538818, 630.2465820
3: -168.2837219, 608.8666992, -164.7066956, 594.9381104, -763.2218018, 773.5733032
4: -143.2449951, 557.9903564, -140.2548218, 544.5361328, -687.7810669, 698.2451172

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9847058, upper bound: 554.9835591
time: 0.96 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9845376, upper bound: 554.9834941
time: 0.91 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -134.9726562, 424.7419739, -134.9726562, 424.7419739, -559.7145996, 559.7145996
1: -188.6135559, 428.4528198, -188.6135559, 428.4528198, -617.0664062, 617.0664062
2: -159.2701263, 474.2145996, -159.2701263, 474.2145996, -633.4847412, 633.4847412
3: -168.2837219, 608.8666992, -168.2837219, 608.8666992, -777.1503906, 777.1503906
4: -143.2449951, 557.9903564, -143.2449951, 557.9903564, -701.2352905, 701.2352905

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9847058, upper bound: 554.9841666
time: 1.20 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9845376, upper bound: 554.9838490
time: 1.70 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -134.9726562, 424.7419739, -138.0580139, 439.1935120, -574.1661377, 562.7999878
1: -188.6135559, 428.4528198, -194.1572723, 441.8686218, -630.4821167, 622.6101074
2: -159.2701263, 474.2145996, -163.9659271, 488.6260071, -647.8959351, 638.1805420
3: -168.2837219, 608.8666992, -173.0027618, 627.9880371, -796.2717285, 781.8693848
4: -143.2449951, 557.9903564, -147.3025970, 573.9549561, -717.1998901, 705.2929688

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9784002, upper bound: 554.9733588
time: 1.12 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9782350, upper bound: 554.9733151
time: 0.93 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -134.9726562, 424.7419739, -141.6154480, 449.2855530, -584.2581787, 566.3574219
1: -188.6135559, 428.4528198, -198.6038208, 452.3257751, -640.9393311, 627.0566406
2: -159.2701263, 474.2145996, -167.6586151, 500.4990234, -659.7690430, 641.8732300
3: -168.2837219, 608.8666992, -177.0885010, 643.3457031, -811.6293945, 785.9550781
4: -143.2449951, 557.9903564, -150.7000732, 588.6919556, -731.9368896, 708.6903687

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9784003, upper bound: 554.9743671
time: 1.13 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9782350, upper bound: 554.9740684
time: 1.10 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -140.7320251, 446.1545105, -128.4147644, 403.8234253, -544.5554199, 574.5692749
1: -197.3291779, 449.2398987, -179.2112732, 407.4659729, -604.7951660, 628.4511719
2: -166.5858002, 497.0308228, -151.4848175, 450.8223877, -617.4082031, 648.5156250
3: -175.9496460, 638.9199219, -159.9417114, 578.7056274, -754.6552124, 798.8616333
4: -149.7327728, 584.6864624, -136.3511658, 529.6724854, -679.4052734, 721.0375977

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9750929, upper bound: 554.9769051
time: 1.46 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9750930, upper bound: 554.9782156
time: 1.00 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -141.6680908, 449.4027405, -127.9205246, 401.9008789, -543.5689697, 577.3232422
1: -198.6617737, 452.4513550, -179.0597534, 405.8210449, -604.4827881, 631.5111084
2: -167.7134552, 500.6373596, -151.2300415, 449.0902405, -616.8035889, 651.8672485
3: -177.1431580, 643.5125732, -159.6991577, 576.3389893, -753.4820557, 803.2117310
4: -150.7537537, 588.8465576, -136.0447540, 527.7747192, -678.5283813, 724.8911743

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9752871, upper bound: 554.9782350
time: 1.17 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9752871, upper bound: 554.9782350
time: 1.06 seconds

## BFS IS instance: IS_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -139.2046967, 440.9418640, -136.1399231, 433.0962524, -572.3008423, 577.0817871
1: -194.6376648, 444.0958252, -191.3532562, 435.8071899, -630.4447632, 635.4490967
2: -164.4753876, 491.3377380, -161.5688324, 481.8968506, -646.3721313, 652.9065552
3: -173.6896820, 631.5960083, -170.5144501, 619.5531616, -793.2427979, 802.1104736
4: -147.9734955, 577.8323364, -145.1862793, 566.1088257, -714.0823364, 723.0184937

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708104, upper bound: 554.9678382
time: 1.47 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708104, upper bound: 554.9692730
time: 1.16 seconds

## BFS IS instance: IS_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -138.9383545, 439.6115417, -137.0725098, 436.2905273, -575.2288818, 576.6840820
1: -194.6855316, 443.0742798, -192.6868896, 438.9768982, -633.6624146, 635.7611694
2: -164.3618774, 490.2503662, -162.6926117, 485.4533997, -649.8152466, 652.9429932
3: -173.6382446, 630.0073853, -171.7039490, 624.0643921, -797.7026367, 801.7113037
4: -147.8123169, 576.6052856, -146.1962280, 570.2287598, -718.0409546, 722.8015137

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699589, upper bound: 554.9699590
time: 0.99 seconds

## Relational analysis of IS_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9699589, upper bound: 554.9699590
time: 1.03 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.34 seconds
IS_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9706410, upper bound: 554.9718688
IS_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9706410, upper bound: 554.9718688
IS_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9706410, upper bound: 554.9718688
IS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9706410, upper bound: 554.9718688
IS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9756812, upper bound: 554.9781013
IS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9702367, upper bound: 554.9702367
IS_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9781013, upper bound: 554.9756881
IS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9702367, upper bound: 554.9702367
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9746680, upper bound: 554.9794736
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9746340, upper bound: 554.9792793
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9746680, upper bound: 554.9794736
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9746340, upper bound: 554.9792793
IS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9700542, upper bound: 554.9714536
IS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9683134, upper bound: 554.9707864
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9688047, upper bound: 554.9716982
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9683134, upper bound: 554.9707864
IS_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9847058, upper bound: 554.9835591
IS_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9845376, upper bound: 554.9834941
IS_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9847058, upper bound: 554.9841666
IS_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9845376, upper bound: 554.9838490
IS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9784002, upper bound: 554.9733588
IS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9782350, upper bound: 554.9733151
IS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9784003, upper bound: 554.9743671
IS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9782350, upper bound: 554.9740684
IS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9750929, upper bound: 554.9769051
IS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9750930, upper bound: 554.9782156
IS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9752871, upper bound: 554.9782350
IS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9752871, upper bound: 554.9782350
IS_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9708104, upper bound: 554.9678382
IS_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9708104, upper bound: 554.9692730
IS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9699589, upper bound: 554.9699590
IS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 0, lower bound: -554.9699589, upper bound: 554.9699590

## BFS IS instance: IS_A1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -130.8796387, 412.2967529, -129.5068207, 407.2422180, -538.1218262, 541.8035889
1: -183.3904114, 415.7405396, -180.8375549, 410.8070984, -594.1975098, 596.5779419
2: -154.8859406, 459.9907227, -152.8676300, 454.5437622, -609.4296875, 612.8583374
3: -163.4978790, 590.3821411, -161.3594513, 583.3352661, -746.8331299, 751.7414551
4: -139.2274323, 540.3772583, -137.5561218, 533.8437500, -673.0711670, 677.9332886

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9788454, upper bound: 554.9773035
time: 1.41 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9788454, upper bound: 554.9773035
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -130.8796387, 412.2967529, -135.6968231, 430.7904663, -561.6701050, 547.9934692
1: -183.3904114, 415.7405396, -190.1970978, 433.5797424, -616.9701538, 605.9375610
2: -154.8859406, 459.9907227, -160.7775116, 479.4364319, -634.3223877, 620.7681274
3: -163.4978790, 590.3821411, -169.6004028, 616.1744995, -779.6723633, 759.9824829
4: -139.2274323, 540.3772583, -144.5901337, 563.0349731, -702.2623901, 684.9672241

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9788454, upper bound: 554.9773035
time: 0.90 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9788454, upper bound: 554.9773035
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -137.1628418, 436.1359253, -129.5068207, 407.2422180, -544.4050293, 565.6426392
1: -192.8791199, 438.8210144, -180.8375549, 410.8070984, -603.6861572, 619.6583862
2: -162.8877106, 485.2140198, -152.8676300, 454.5437622, -617.4314575, 638.0816650
3: -171.8616943, 623.6537476, -161.3594513, 583.3352661, -755.1969604, 785.0131226
4: -146.3327789, 570.0169067, -137.5561218, 533.8437500, -680.1765137, 707.5729980

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9706410, upper bound: 554.9718688
time: 1.07 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9706410, upper bound: 554.9718688
time: 3.43 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -137.1628418, 436.1359253, -135.6968231, 430.7904663, -567.9533081, 571.8325806
1: -192.8791199, 438.8210144, -190.1970978, 433.5797424, -626.4588013, 629.0179443
2: -162.8877106, 485.2140198, -160.7775116, 479.4364319, -642.3241577, 645.9915161
3: -171.8616943, 623.6537476, -169.6004028, 616.1744995, -788.0361938, 793.2541504
4: -146.3327789, 570.0169067, -144.5901337, 563.0349731, -709.3677368, 714.6070557

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9706410, upper bound: 554.9718688
time: 1.17 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9706410, upper bound: 554.9718688
time: 1.40 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -140.8496704, 446.2448425, -129.1332550, 405.8225708, -546.6722412, 575.3781128
1: -197.3220520, 449.2522888, -180.8202820, 409.6738281, -606.9958496, 630.0725708
2: -166.8048248, 496.8097229, -152.7237396, 453.3137512, -620.1185913, 649.5334473
3: -175.9925079, 638.1691284, -161.2488556, 581.6445312, -757.6370239, 799.4179077
4: -150.0195465, 583.3884277, -137.3544464, 532.4979858, -682.5175171, 720.7427979

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718688, upper bound: 554.9706410
time: 1.05 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718688, upper bound: 554.9706410
time: 1.39 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -139.8300323, 443.4956665, -135.3625336, 429.4769592, -569.3068848, 578.8581543
1: -195.9462128, 446.4235229, -190.2272034, 432.5570984, -628.5032959, 636.6507568
2: -165.6338043, 493.6914673, -160.6483917, 478.3566895, -643.9904785, 654.3398438
3: -174.7569427, 634.2959595, -169.5371857, 614.6412354, -789.3981323, 803.8331299
4: -148.9625854, 579.7820435, -144.3938904, 561.8070068, -710.7695923, 724.1758423

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718688, upper bound: 554.9706410
time: 1.37 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9718688, upper bound: 554.9706410
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -129.1332550, 405.8225708, -140.3836212, 444.4812622, -573.6144409, 546.2061157
1: -180.8202820, 409.6738281, -197.1487579, 447.8031921, -628.6234741, 606.8225708
2: -152.7237396, 453.3137512, -166.5076141, 495.2358704, -647.9595947, 619.8213501
3: -161.2488556, 581.6445312, -175.7447510, 636.0303345, -797.2791748, 757.3892822
4: -137.3544464, 532.4979858, -149.6789398, 581.5927124, -718.9470215, 682.1768799

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9702367, upper bound: 554.9702367
time: 1.06 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9702367, upper bound: 554.9702367
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -135.3625336, 429.4769592, -139.3867188, 441.8024902, -577.1650391, 568.8636475
1: -190.2272034, 432.5570984, -195.8007507, 445.0257263, -635.2529297, 628.3578491
2: -160.6483917, 478.3566895, -165.3587646, 492.1918030, -652.8402100, 643.7153320
3: -169.5371857, 614.6412354, -174.5352173, 632.2255249, -801.7626953, 789.1764526
4: -144.3938904, 561.8070068, -148.6382599, 578.0553589, -722.4492188, 710.4452515

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9702367, upper bound: 554.9702367
time: 1.23 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9702367, upper bound: 554.9702367
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -130.8796387, 412.2967529, -133.0537109, 418.0853271, -548.9649658, 545.3504028
1: -183.3904114, 415.7405396, -185.2976227, 421.8851013, -605.2754517, 601.0380249
2: -154.8859406, 459.9907227, -156.6186371, 466.9302368, -621.8161621, 616.6092529
3: -163.4978790, 590.3821411, -165.4773560, 599.5919189, -763.0897827, 755.8594360
4: -139.2274323, 540.3772583, -140.9972992, 549.2918701, -688.5192871, 681.3745117

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9834941, upper bound: 554.9855352
time: 1.12 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9834941, upper bound: 554.9855532
time: 1.39 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -131.8264923, 415.5336609, -132.3233948, 415.1637573, -546.9902344, 547.8570557
1: -184.7457886, 418.9423218, -184.7430267, 419.2759399, -604.0217285, 603.6853027
2: -156.0319977, 463.5839844, -156.0207367, 464.0734558, -620.1054688, 619.6047363
3: -164.7066956, 594.9381104, -164.8735657, 595.7244873, -760.4310913, 759.8114624
4: -140.2548218, 544.5361328, -140.3941803, 546.0523071, -686.3070068, 684.9302368

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9834941, upper bound: 554.9855352
time: 1.46 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9834941, upper bound: 554.9855532
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -137.1628418, 436.1359253, -133.0537109, 418.0853271, -555.2481689, 569.1895142
1: -192.8791199, 438.8210144, -185.2976227, 421.8851013, -614.7640991, 624.1185303
2: -162.8877106, 485.2140198, -156.6186371, 466.9302368, -629.8179321, 641.8326416
3: -171.8616943, 623.6537476, -165.4773560, 599.5919189, -771.4536133, 789.1311035
4: -146.3327789, 570.0169067, -140.9972992, 549.2918701, -695.6246338, 711.0142212

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9746326, upper bound: 554.9792793
time: 1.18 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9746326, upper bound: 554.9792793
time: 1.28 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -138.0580139, 439.1935120, -132.3233948, 415.1637573, -553.2218018, 571.5169067
1: -194.1572723, 441.8686218, -184.7430267, 419.2759399, -613.4331665, 626.6116333
2: -163.9659271, 488.6260071, -156.0207367, 464.0734558, -628.0393677, 644.6467285
3: -173.0027618, 627.9880371, -164.8735657, 595.7244873, -768.7271729, 792.8614502
4: -147.3025970, 573.9549561, -140.3941803, 546.0523071, -693.3549194, 714.3491211

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9746340, upper bound: 554.9792793
time: 1.12 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9746340, upper bound: 554.9792793
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -129.5068207, 407.2422180, -140.7320251, 446.1545105, -575.6613159, 547.9742432
1: -180.8375549, 410.8070984, -197.3291779, 449.2398987, -630.0773926, 608.1362915
2: -152.8676300, 454.5437622, -166.5858002, 497.0308228, -649.8984375, 621.1295166
3: -161.3594513, 583.3352661, -175.9496460, 638.9199219, -800.2792358, 759.2848511
4: -137.5561218, 533.8437500, -149.7327728, 584.6864624, -722.2424927, 683.5764160

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9769051, upper bound: 554.9750051
time: 1.15 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9769051, upper bound: 554.9750051
time: 1.45 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -129.1332550, 405.8225708, -141.6680908, 449.4027405, -578.5360107, 547.4905396
1: -180.8202820, 409.6738281, -198.6617737, 452.4513550, -633.2716064, 608.3355713
2: -152.7237396, 453.3137512, -167.7134552, 500.6373596, -653.3610229, 621.0272217
3: -161.2488556, 581.6445312, -177.1431580, 643.5125732, -804.7614136, 758.7875977
4: -137.3544464, 532.4979858, -150.7537537, 588.8465576, -726.2009277, 683.2517090

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9769051, upper bound: 554.9750051
time: 0.93 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9769051, upper bound: 554.9750051
time: 1.54 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -137.1628418, 436.1359253, -139.2046967, 440.9418640, -578.1047363, 575.3403931
1: -192.8791199, 438.8210144, -194.6376648, 444.0958252, -636.9748535, 633.4584961
2: -162.8877106, 485.2140198, -164.4753876, 491.3377380, -654.2254639, 649.6893311
3: -171.8616943, 623.6537476, -173.6896820, 631.5960083, -803.4577026, 797.3433838
4: -146.3327789, 570.0169067, -147.9734955, 577.8323364, -724.1651001, 717.9904175

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683134, upper bound: 554.9707864
time: 1.51 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683134, upper bound: 554.9707864
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -138.0580139, 439.1935120, -138.9383545, 439.6115417, -577.6695557, 578.1318359
1: -194.1572723, 441.8686218, -194.6855316, 443.0742798, -637.2314453, 636.5541382
2: -163.9659271, 488.6260071, -164.3618774, 490.2503662, -654.2163086, 652.9877319
3: -173.0027618, 627.9880371, -173.6382446, 630.0073853, -803.0100708, 801.6262207
4: -147.3025970, 573.9549561, -147.8123169, 576.6052856, -723.9078979, 721.7672119

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683134, upper bound: 554.9707864
time: 1.22 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683134, upper bound: 554.9707864
time: 1.08 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -133.0537109, 418.0853271, -130.8796387, 412.2967529, -545.3504028, 548.9649658
1: -185.2976227, 421.8851013, -183.3904114, 415.7405396, -601.0380249, 605.2754517
2: -156.6186371, 466.9302368, -154.8859406, 459.9907227, -616.6092529, 621.8161621
3: -165.4773560, 599.5919189, -163.4978790, 590.3821411, -755.8594360, 763.0897827
4: -140.9972992, 549.2918701, -139.2274323, 540.3772583, -681.3745117, 688.5192871

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855352, upper bound: 554.9834941
time: 1.30 seconds

## Relational analysis of IS_A2_A1_B1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855352, upper bound: 554.9834941
time: 0.97 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -132.3233948, 415.1637573, -131.8264923, 415.5336609, -547.8570557, 546.9902344
1: -184.7430267, 419.2759399, -184.7457886, 418.9423218, -603.6853027, 604.0217285
2: -156.0207367, 464.0734558, -156.0319977, 463.5839844, -619.6046753, 620.1054688
3: -164.8735657, 595.7244873, -164.7066956, 594.9381104, -759.8114624, 760.4310913
4: -140.3941803, 546.0523071, -140.2548218, 544.5361328, -684.9302368, 686.3070068

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855352, upper bound: 554.9834941
time: 1.05 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9855352, upper bound: 554.9834941
time: 1.16 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -133.0537109, 418.0853271, -134.0399933, 421.5050659, -554.5587769, 552.1253052
1: -185.2976227, 421.8851013, -187.2750854, 425.2625427, -610.5601807, 609.1600342
2: -156.6186371, 466.9302368, -158.1349030, 470.6487427, -627.2673950, 625.0651245
3: -165.4773560, 599.5919189, -167.0919952, 604.3105469, -769.7878418, 766.6837769
4: -140.9972992, 549.2918701, -142.2231140, 553.8601685, -694.8574829, 691.5150146

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9845376, upper bound: 554.9838490
time: 1.28 seconds

## Relational analysis of IS_A2_A1_B1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9845376, upper bound: 554.9838490
time: 1.25 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -132.3233948, 415.1637573, -134.9726562, 424.7419739, -557.0653687, 550.1364136
1: -184.7430267, 419.2759399, -188.6135559, 428.4528198, -613.1958618, 607.8894653
2: -156.0207367, 464.0734558, -159.2701263, 474.2145996, -630.2353516, 623.3435059
3: -164.8735657, 595.7244873, -168.2837219, 608.8666992, -773.7401123, 764.0081787
4: -140.3941803, 546.0523071, -143.2449951, 557.9903564, -698.3844604, 689.2971802

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9845376, upper bound: 554.9838490
time: 1.54 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2_B2

### Relational analysis result of IS_A2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9845376, upper bound: 554.9838490
time: 1.13 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.26 seconds
IS_A1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9788454, upper bound: 554.9773035
IS_A1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9788454, upper bound: 554.9773035
IS_A1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9788454, upper bound: 554.9773035
IS_A1_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9788454, upper bound: 554.9773035
IS_A1_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9706410, upper bound: 554.9718688
IS_A1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9706410, upper bound: 554.9718688
IS_A1_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9706410, upper bound: 554.9718688
IS_A1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9706410, upper bound: 554.9718688
IS_A1_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9718688, upper bound: 554.9706410
IS_A1_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9718688, upper bound: 554.9706410
IS_A1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9718688, upper bound: 554.9706410
IS_A1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9718688, upper bound: 554.9706410
IS_A1_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9702367, upper bound: 554.9702367
IS_A1_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9702367, upper bound: 554.9702367
IS_A1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9702367, upper bound: 554.9702367
IS_A1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9702367, upper bound: 554.9702367
IS_A1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9834941, upper bound: 554.9855352
IS_A1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9834941, upper bound: 554.9855532
IS_A1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9834941, upper bound: 554.9855352
IS_A1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9834941, upper bound: 554.9855532
IS_A1_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9746326, upper bound: 554.9792793
IS_A1_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9746326, upper bound: 554.9792793
IS_A1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9746340, upper bound: 554.9792793
IS_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9746340, upper bound: 554.9792793
IS_A1_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9769051, upper bound: 554.9750051
IS_A1_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9769051, upper bound: 554.9750051
IS_A1_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9769051, upper bound: 554.9750051
IS_A1_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9769051, upper bound: 554.9750051
IS_A1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9683134, upper bound: 554.9707864
IS_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9683134, upper bound: 554.9707864
IS_A1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9683134, upper bound: 554.9707864
IS_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9683134, upper bound: 554.9707864
IS_A2_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9855352, upper bound: 554.9834941
IS_A2_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9855352, upper bound: 554.9834941
IS_A2_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9855352, upper bound: 554.9834941
IS_A2_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9855352, upper bound: 554.9834941
IS_A2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9845376, upper bound: 554.9838490
IS_A2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9845376, upper bound: 554.9838490
IS_A2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9845376, upper bound: 554.9838490
IS_A2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.26
Output dim: 0, lower bound: -554.9845376, upper bound: 554.9838490
IS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 0, lower bound: -554.9784002, upper bound: 554.9733588
IS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 0, lower bound: -554.9782350, upper bound: 554.9733151
IS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 0, lower bound: -554.9784003, upper bound: 554.9743671
IS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 0, lower bound: -554.9782350, upper bound: 554.9740684
IS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 0, lower bound: -554.9750929, upper bound: 554.9769051
IS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 0, lower bound: -554.9750930, upper bound: 554.9782156
IS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 0, lower bound: -554.9752871, upper bound: 554.9782350
IS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 0, lower bound: -554.9752871, upper bound: 554.9782350
IS_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 0, lower bound: -554.9708104, upper bound: 554.9678382
IS_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 0, lower bound: -554.9708104, upper bound: 554.9692730
IS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 0, lower bound: -554.9699589, upper bound: 554.9699590
IS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.26
Output dim: 0, lower bound: -554.9699589, upper bound: 554.9699590
Binary search (step 4): status=Status.UNKNOWN, low=0.1875000, high=0.2187500, mid=0.2187500, abs_max=619.6494140625
rel_dist={0: [-554.9906328569723, 554.9906328569723]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.1875
execution time: 1107.72 seconds
