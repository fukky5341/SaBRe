## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_5.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 27.5202488034


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706)
1: (-6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965)
2: (-5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000)
3: (-7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002)
4: (-5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062)

## BASE Result
execution time: IAR + LP analysis = 2.63 + 2.01 = 4.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -27.5477966, upper bound: 27.5477966


# Binary Search by BASE starts (time budget: 1195.37 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=31.572500228881836
rel_dist={3: [-27.547796646245764, 27.54779664624577]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=31.572500228881836
rel_dist={3: [-27.545485351818588, 27.545485351818584]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=31.572500228881836
rel_dist={3: [-27.542158575057197, 27.5421585750572]}

## Binary search (step 3) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=31.572500228881836
rel_dist={3: [-27.539737497385985, 27.53973749739012]}

## Binary search (step 4) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=31.572500228881836
rel_dist={3: [-27.537829206615235, 27.53782920661805]}

## Binary search (step 5) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=31.572500228881836
rel_dist={3: [-27.536721022153646, 27.536721022157913]}

## Binary search (step 6) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=31.572500228881836
rel_dist={3: [-27.53614770554994, 27.536147760020427]}

## Binary search (step 7) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=31.572500228881836
rel_dist={3: [-27.535855068265125, 27.535855069703167]}

## Binary search (step 8) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=31.572500228881836
rel_dist={3: [-27.535707377583634, 27.535707377584167]}

## Binary search (step 9) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=31.572500228881836
rel_dist={3: [-27.53563353305055, 27.535633533050827]}

## Binary search (step 10) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=31.572500228881836
rel_dist={3: [-27.53559661294977, 27.53559660993387]}

## Binary search (step 11) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=31.572500228881836
rel_dist={3: [-27.535578156170903, 27.53557815617097]}

## Binary search (step 12) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=31.572500228881836
rel_dist={3: [-27.53556893416836, 27.535568931060084]}

## Binary search (step 13) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=31.572500228881836
rel_dist={3: [-27.535564296056833, 27.535564333130978]}

## Binary search (step 14) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=31.572500228881836
rel_dist={3: [-27.535561987225833, 27.535562203225297]}

## Binary search (step 15) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=31.572500228881836
rel_dist={3: [-27.53556094347148, 27.535560966559956]}

## Binary search (step 16) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=31.572500228881836
rel_dist={3: [-27.53556054532261, 27.53556089300605]}

## Binary Search Result
Binary search time: 74.11 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1121.26 seconds

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5300102, upper bound: 27.5386763
time: 0.83 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5457634, upper bound: 27.5457635
time: 0.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.63 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.63
Output dim: 3, lower bound: -27.5300102, upper bound: 27.5386763
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.63
Output dim: 3, lower bound: -27.5457634, upper bound: 27.5457635

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.4779983, 15.6032467, -4.8926516, 16.6545238, -21.1325207, 20.4958973
1: -6.4072104, 15.9771137, -6.9941020, 17.1100998, -23.5173073, 22.9712143
2: -5.4055552, 17.9303551, -5.9263554, 19.1902447, -24.5958004, 23.8567104
3: -6.4871049, 22.9893456, -7.0618343, 24.5106659, -30.9977703, 30.0511799
4: -5.3025470, 21.2900162, -5.7611084, 22.7441978, -28.0467453, 27.0511246

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5229229, upper bound: 27.5229229
time: 0.65 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5229229, upper bound: 27.5386763
time: 0.79 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -4.8002801, 16.4875641, -4.8926516, 16.6545238, -21.4548035, 21.3802128
1: -6.8583674, 16.9091358, -6.9941020, 17.1100998, -23.9684658, 23.9032364
2: -5.7989798, 18.9445438, -5.9263554, 19.1902447, -24.9892235, 24.8708992
3: -6.9374380, 24.2584801, -7.0618343, 24.5106659, -31.4481030, 31.3203144
4: -5.6505985, 22.4603367, -5.7611084, 22.7441978, -28.3947964, 28.2214451

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5386763, upper bound: 27.5300102
time: 0.71 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5386763, upper bound: 27.5457634
time: 0.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.90 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.90
Output dim: 3, lower bound: -27.5229229, upper bound: 27.5229229
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.90
Output dim: 3, lower bound: -27.5229229, upper bound: 27.5386763
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.90
Output dim: 3, lower bound: -27.5386763, upper bound: 27.5300102
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.90
Output dim: 3, lower bound: -27.5386763, upper bound: 27.5457634

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -4.4779983, 15.6032467, -4.4779983, 15.6032467, -20.0812454, 20.0812454
1: -6.4072104, 15.9771137, -6.4072104, 15.9771137, -22.3843231, 22.3843231
2: -5.4055552, 17.9303551, -5.4055552, 17.9303551, -23.3359108, 23.3359108
3: -6.4871049, 22.9893456, -6.4871049, 22.9893456, -29.4764500, 29.4764500
4: -5.3025470, 21.2900162, -5.3025470, 21.2900162, -26.5925636, 26.5925636

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5218199, upper bound: 27.5193727
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5229229, upper bound: 27.5229229
time: 1.07 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -4.4779983, 15.6032467, -4.8002801, 16.4875641, -20.9655628, 20.4035263
1: -6.4072104, 15.9771137, -6.8583674, 16.9091358, -23.3163452, 22.8354797
2: -5.4055552, 17.9303551, -5.7989798, 18.9445438, -24.3500996, 23.7293358
3: -6.4871049, 22.9893456, -6.9374380, 24.2584801, -30.7455845, 29.9267845
4: -5.3025470, 21.2900162, -5.6505985, 22.4603367, -27.7628841, 26.9406128

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5218199, upper bound: 27.5351261
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5229229, upper bound: 27.5386763
time: 0.74 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -4.8002801, 16.4875641, -4.4779983, 15.6032467, -20.4035263, 20.9655628
1: -6.8583674, 16.9091358, -6.4072104, 15.9771137, -22.8354816, 23.3163452
2: -5.7989798, 18.9445438, -5.4055552, 17.9303551, -23.7293358, 24.3500996
3: -6.9374380, 24.2584801, -6.4871049, 22.9893456, -29.9267845, 30.7455845
4: -5.6505985, 22.4603367, -5.3025470, 21.2900162, -26.9406128, 27.7628841

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378310, upper bound: 27.5262572
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5386762, upper bound: 27.5299076
time: 0.88 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -4.8002801, 16.4875641, -4.8002801, 16.4875641, -21.2878418, 21.2878418
1: -6.8583674, 16.9091358, -6.8583674, 16.9091358, -23.7675037, 23.7675018
2: -5.7989798, 18.9445438, -5.7989798, 18.9445438, -24.7435226, 24.7435226
3: -6.9374380, 24.2584801, -6.9374380, 24.2584801, -31.1959171, 31.1959171
4: -5.6505985, 22.4603367, -5.6505985, 22.4603367, -28.1109352, 28.1109352

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378311, upper bound: 27.5420105
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5386763, upper bound: 27.5456609
time: 0.81 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.54 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 3, lower bound: -27.5218199, upper bound: 27.5193727
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 3, lower bound: -27.5229229, upper bound: 27.5229229
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 3, lower bound: -27.5218199, upper bound: 27.5351261
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 3, lower bound: -27.5229229, upper bound: 27.5386763
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 3, lower bound: -27.5378310, upper bound: 27.5262572
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 3, lower bound: -27.5386762, upper bound: 27.5299076
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 3, lower bound: -27.5378311, upper bound: 27.5420105
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 3, lower bound: -27.5386763, upper bound: 27.5456609

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.8096104, 13.5872917, -4.4779983, 15.6032467, -19.4128571, 18.0652905
1: -5.5293055, 13.9026499, -6.4072104, 15.9771137, -21.5064201, 20.3098602
2: -4.6612096, 15.6515274, -5.4055552, 17.9303551, -22.5915642, 21.0570812
3: -5.5544548, 20.0211716, -6.4871049, 22.9893456, -28.5438004, 26.5082760
4: -4.5881572, 18.5381012, -5.3025470, 21.2900162, -25.8781719, 23.8406487

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5182697, upper bound: 27.5182697
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5182697, upper bound: 27.5193727
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.3976188, 15.3468742, -4.4779983, 15.6032467, -20.0008659, 19.8248711
1: -6.2940626, 15.7216339, -6.4072104, 15.9771137, -22.2711735, 22.1288433
2: -5.3095002, 17.6520042, -5.4055552, 17.9303551, -23.2398548, 23.0575581
3: -6.3730483, 22.6263657, -6.4871049, 22.9893456, -29.3623943, 29.1134701
4: -5.2157865, 20.9694633, -5.3025470, 21.2900162, -26.5058022, 26.2720089

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5193727, upper bound: 27.5218199
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5193727, upper bound: 27.5229229
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.8096104, 13.5872917, -4.8002801, 16.4875641, -20.2971725, 18.3875713
1: -5.5293055, 13.9026499, -6.8583674, 16.9091358, -22.4384422, 20.7610168
2: -4.6612096, 15.6515274, -5.7989798, 18.9445438, -23.6057529, 21.4505081
3: -5.5544548, 20.0211716, -6.9374380, 24.2584801, -29.8129349, 26.9586105
4: -4.5881572, 18.5381012, -5.6505985, 22.4603367, -27.0484943, 24.1886978

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5182697, upper bound: 27.5182697
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5251542, upper bound: 27.5351259
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.3976188, 15.3468742, -4.8002801, 16.4875641, -20.8851814, 20.1471519
1: -6.2940626, 15.7216339, -6.8583674, 16.9091358, -23.2031956, 22.5799999
2: -5.3095002, 17.6520042, -5.7989798, 18.9445438, -24.2540436, 23.4509850
3: -6.3730483, 22.6263657, -6.9374380, 24.2584801, -30.6315289, 29.5638027
4: -5.2157865, 20.9694633, -5.6505985, 22.4603367, -27.6761227, 26.6200619

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5262572, upper bound: 27.5378311
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5262572, upper bound: 27.5386761
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.0815225, 14.3603363, -4.4779983, 15.6032467, -19.6847687, 18.8383350
1: -5.9201736, 14.7163811, -6.4072104, 15.9771137, -21.8972874, 21.1235886
2: -5.0029421, 16.5449162, -5.4055552, 17.9303551, -22.9332962, 21.9504719
3: -5.9411888, 21.1416397, -6.4871049, 22.9893456, -28.9305344, 27.6287441
4: -4.8949690, 19.5565891, -5.3025470, 21.2900162, -26.1849861, 24.8591347

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5342809, upper bound: 27.5251542
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5342809, upper bound: 27.5262572
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.7206116, 16.2323303, -4.4779983, 15.6032467, -20.3238564, 20.7103252
1: -6.7455416, 16.6543579, -6.4072104, 15.9771137, -22.7226543, 23.0615692
2: -5.7037611, 18.6685505, -5.4055552, 17.9303551, -23.6341133, 24.0741062
3: -6.8226271, 23.8968391, -6.4871049, 22.9893456, -29.8119736, 30.3839436
4: -5.5644245, 22.1408634, -5.3025470, 21.2900162, -26.8544388, 27.4434109

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5351260, upper bound: 27.5288045
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5351260, upper bound: 27.5299076
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.0815225, 14.3603363, -4.8002801, 16.4875641, -20.5690842, 19.1606159
1: -5.9201736, 14.7163811, -6.8583674, 16.9091358, -22.8293095, 21.5747452
2: -5.0029421, 16.5449162, -5.7989798, 18.9445438, -23.9474831, 22.3438950
3: -5.9411888, 21.1416397, -6.9374380, 24.2584801, -30.1996670, 28.0790768
4: -4.8949690, 19.5565891, -5.6505985, 22.4603367, -27.3553047, 25.2071838

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5411653, upper bound: 27.5411624
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5411653, upper bound: 27.5420104
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.7206116, 16.2323303, -4.8002801, 16.4875641, -21.2081718, 21.0326080
1: -6.7455416, 16.6543579, -6.8583674, 16.9091358, -23.6546783, 23.5127258
2: -5.7037611, 18.6685505, -5.7989798, 18.9445438, -24.6483021, 24.4675293
3: -6.8226271, 23.8968391, -6.9374380, 24.2584801, -31.0811062, 30.8342743
4: -5.5644245, 22.1408634, -5.6505985, 22.4603367, -28.0247612, 27.7914619

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5419888, upper bound: 27.5448052
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5419888, upper bound: 27.5456608
time: 0.74 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.98 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.98
Output dim: 3, lower bound: -27.5182697, upper bound: 27.5182697
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.98
Output dim: 3, lower bound: -27.5182697, upper bound: 27.5193727
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -27.5193727, upper bound: 27.5218199
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -27.5193727, upper bound: 27.5229229
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.98
Output dim: 3, lower bound: -27.5182697, upper bound: 27.5182697
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -27.5251542, upper bound: 27.5351259
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -27.5262572, upper bound: 27.5378311
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -27.5262572, upper bound: 27.5386761
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -27.5342809, upper bound: 27.5251542
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -27.5342809, upper bound: 27.5262572
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -27.5351260, upper bound: 27.5288045
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -27.5351260, upper bound: 27.5299076
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -27.5411653, upper bound: 27.5411624
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -27.5411653, upper bound: 27.5420104
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -27.5419888, upper bound: 27.5448052
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -27.5419888, upper bound: 27.5456608

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.3976188, 15.3468742, -3.8096104, 13.5872917, -17.9849110, 19.1564846
1: -6.2940626, 15.7216339, -5.5293055, 13.9026499, -20.1967125, 21.2509384
2: -5.3095002, 17.6520042, -4.6612096, 15.6515274, -20.9610271, 22.3132133
3: -6.3730483, 22.6263657, -5.5544548, 20.0211716, -26.3942204, 28.1808205
4: -5.2157865, 20.9694633, -4.5881572, 18.5381012, -23.7538872, 25.5576191

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4958722, upper bound: 27.4984862
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4826532, upper bound: 27.4851750
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.3976188, 15.3468742, -4.3976188, 15.3468742, -19.7444916, 19.7444916
1: -6.2940626, 15.7216339, -6.2940626, 15.7216339, -22.0156937, 22.0156937
2: -5.3095002, 17.6520042, -5.3095002, 17.6520042, -22.9615021, 22.9615021
3: -6.3730483, 22.6263657, -6.3730483, 22.6263657, -28.9994144, 28.9994144
4: -5.2157865, 20.9694633, -5.2157865, 20.9694633, -26.1852493, 26.1852493

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4958722, upper bound: 27.4986913
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4828583, upper bound: 27.4851750
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.8096104, 13.5872917, -4.7206116, 16.2323303, -20.0419388, 18.3079014
1: -5.5293055, 13.9026499, -6.7455416, 16.6543579, -22.1836624, 20.6481915
2: -4.6612096, 15.6515274, -5.7037611, 18.6685505, -23.3297596, 21.3552837
3: -5.5544548, 20.0211716, -6.8226271, 23.8968391, -29.4512939, 26.8437996
4: -4.5881572, 18.5381012, -5.5644245, 22.1408634, -26.7290192, 24.1025238

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5144679, upper bound: 27.5286491
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5045526, upper bound: 27.5184922
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.3976188, 15.3468742, -4.0815225, 14.3603363, -18.7579556, 19.4283924
1: -6.2940626, 15.7216339, -5.9201736, 14.7163811, -21.0104370, 21.6418076
2: -5.3095002, 17.6520042, -5.0029421, 16.5449162, -21.8544159, 22.6549435
3: -6.3730483, 22.6263657, -5.9411888, 21.1416397, -27.5146885, 28.5675526
4: -5.2157865, 20.9694633, -4.8949690, 19.5565891, -24.7723751, 25.8644333

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5177716, upper bound: 27.5337204
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5047577, upper bound: 27.5204090
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.3976188, 15.3468742, -4.7206116, 16.2323303, -20.6299438, 20.0674801
1: -6.2940626, 15.7216339, -6.7455416, 16.6543579, -22.9484215, 22.4671745
2: -5.3095002, 17.6520042, -5.7037611, 18.6685505, -23.9780502, 23.3557606
3: -6.3730483, 22.6263657, -6.8226271, 23.8968391, -30.2698879, 29.4489918
4: -5.2157865, 20.9694633, -5.5644245, 22.1408634, -27.3566494, 26.5338860

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4958722, upper bound: 27.5343254
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4828583, upper bound: 27.5210141
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.0815225, 14.3603363, -3.8096104, 13.5872917, -17.6688137, 18.1699467
1: -5.9201736, 14.7163811, -5.5293055, 13.9026499, -19.8228226, 20.2456818
2: -5.0029421, 16.5449162, -4.6612096, 15.6515274, -20.6544685, 21.2061253
3: -5.9411888, 21.1416397, -5.5544548, 20.0211716, -25.9623604, 26.6960945
4: -4.8949690, 19.5565891, -4.5881572, 18.5381012, -23.4330711, 24.1447430

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5064620, upper bound: 27.4995374
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5178869, upper bound: 27.5045526
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.0815225, 14.3603363, -4.3976188, 15.3468742, -19.4283924, 18.7579556
1: -5.9201736, 14.7163811, -6.2940626, 15.7216339, -21.6418076, 21.0104370
2: -5.0029421, 16.5449162, -5.3095002, 17.6520042, -22.6549435, 21.8544159
3: -5.9411888, 21.1416397, -6.3730483, 22.6263657, -28.5675526, 27.5146885
4: -4.8949690, 19.5565891, -5.2157865, 20.9694633, -25.8644333, 24.7723751

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5064619, upper bound: 27.4997424
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5178870, upper bound: 27.4995373
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.7206116, 16.2323303, -3.8096104, 13.5872917, -18.3079014, 20.0419369
1: -6.7455416, 16.6543579, -5.5293055, 13.9026499, -20.6481915, 22.1836624
2: -5.7037611, 18.6685505, -4.6612096, 15.6515274, -21.3552837, 23.3297596
3: -6.8226271, 23.8968391, -5.5544548, 20.0211716, -26.8437996, 29.4512939
4: -5.5644245, 22.1408634, -4.5881572, 18.5381012, -24.1025238, 26.7290192

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5072745, upper bound: 27.5064950
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5184920, upper bound: 27.5081514
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.7206116, 16.2323303, -4.3976188, 15.3468742, -20.0674801, 20.6299458
1: -6.7455416, 16.6543579, -6.2940626, 15.7216339, -22.4671745, 22.9484215
2: -5.7037611, 18.6685505, -5.3095002, 17.6520042, -23.3557606, 23.9780502
3: -6.8226271, 23.8968391, -6.3730483, 22.6263657, -29.4489918, 30.2698879
4: -5.5644245, 22.1408634, -5.2157865, 20.9694633, -26.5338860, 27.3566494

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5072745, upper bound: 27.5067001
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5184920, upper bound: 27.5081514
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.0815225, 14.3603363, -4.0815225, 14.3603363, -18.4418564, 18.4418564
1: -5.9201736, 14.7163811, -5.9201736, 14.7163811, -20.6365509, 20.6365528
2: -5.0029421, 16.5449162, -5.0029421, 16.5449162, -21.5478573, 21.5478573
3: -5.9411888, 21.1416397, -5.9411888, 21.1416397, -27.0828247, 27.0828266
4: -4.8949690, 19.5565891, -4.8949690, 19.5565891, -24.4515572, 24.4515572

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5257202, upper bound: 27.5267463
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5397864, upper bound: 27.5397819
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.0815225, 14.3603363, -4.7206116, 16.2323303, -20.3138466, 19.0809460
1: -5.9201736, 14.7163811, -6.7455416, 16.6543579, -22.5745316, 21.4619217
2: -5.0029421, 16.5449162, -5.7037611, 18.6685505, -23.6714935, 22.2486744
3: -5.9411888, 21.1416397, -6.8226271, 23.8968391, -29.8380241, 27.9642639
4: -4.8949690, 19.5565891, -5.5644245, 22.1408634, -27.0358315, 25.1210098

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5257201, upper bound: 27.5276985
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5397864, upper bound: 27.5403916
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.7206116, 16.2323303, -4.0815225, 14.3603363, -19.0809479, 20.3138466
1: -6.7455416, 16.6543579, -5.9201736, 14.7163811, -21.4619198, 22.5745316
2: -5.7037611, 18.6685505, -5.0029421, 16.5449162, -22.2486744, 23.6714935
3: -6.8226271, 23.8968391, -5.9411888, 21.1416397, -27.9642639, 29.8380260
4: -5.5644245, 22.1408634, -4.8949690, 19.5565891, -25.1210098, 27.0358315

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5286732, upper bound: 27.5403695
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5403756, upper bound: 27.5433139
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.7206116, 16.2323303, -4.7206116, 16.2323303, -20.9529343, 20.9529324
1: -6.7455416, 16.6543579, -6.7455416, 16.6543579, -23.3998985, 23.3998985
2: -5.7037611, 18.6685505, -5.7037611, 18.6685505, -24.3723106, 24.3723106
3: -6.8226271, 23.8968391, -6.8226271, 23.8968391, -30.7194633, 30.7194653
4: -5.5644245, 22.1408634, -5.5644245, 22.1408634, -27.7052860, 27.7052860

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5286731, upper bound: 27.5414444
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5403756, upper bound: 27.5439897
time: 0.67 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.11 seconds
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.4958722, upper bound: 27.4984862
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.4826532, upper bound: 27.4851750
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.4958722, upper bound: 27.4986913
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.4828583, upper bound: 27.4851750
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5144679, upper bound: 27.5286491
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5045526, upper bound: 27.5184922
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5177716, upper bound: 27.5337204
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5047577, upper bound: 27.5204090
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.4958722, upper bound: 27.5343254
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.4828583, upper bound: 27.5210141
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5064620, upper bound: 27.4995374
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5178869, upper bound: 27.5045526
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5064619, upper bound: 27.4997424
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5178870, upper bound: 27.4995373
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5072745, upper bound: 27.5064950
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5184920, upper bound: 27.5081514
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5072745, upper bound: 27.5067001
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5184920, upper bound: 27.5081514
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5257202, upper bound: 27.5267463
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5397864, upper bound: 27.5397819
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5257201, upper bound: 27.5276985
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5397864, upper bound: 27.5403916
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5286732, upper bound: 27.5403695
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5403756, upper bound: 27.5433139
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5286731, upper bound: 27.5414444
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5403756, upper bound: 27.5439897

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.5618644, 12.9171572, -4.7206116, 16.2323303, -19.7941895, 17.6377659
1: -5.1606417, 13.1869688, -6.7455416, 16.6543579, -21.8149986, 19.9325104
2: -4.3412910, 14.8576298, -5.7037611, 18.6685505, -23.0098419, 20.5613899
3: -5.1961222, 19.0373878, -6.8226271, 23.8968391, -29.0929604, 25.8600121
4: -4.2972727, 17.5959225, -5.5644245, 22.1408634, -26.4381371, 23.1603451

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5064951, upper bound: 27.5072746
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5064951, upper bound: 27.5072746
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.1293182, 14.6288443, -4.0815225, 14.3603363, -18.4896545, 18.7103653
1: -5.9055848, 14.9554243, -5.9201736, 14.7163811, -20.6219616, 20.8755989
2: -4.9716229, 16.8073635, -5.0029421, 16.5449162, -21.5165367, 21.8103046
3: -5.9947290, 21.5765266, -5.9411888, 21.1416397, -27.1363678, 27.5177155
4: -4.9104395, 19.9661102, -4.8949690, 19.5565891, -24.4670258, 24.8610802

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4997424, upper bound: 27.5089837
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4997424, upper bound: 27.5089839
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.5115433, 15.8061476, -4.0815225, 14.3603363, -18.8718796, 19.8876686
1: -6.4535074, 16.1446762, -5.9201736, 14.7163811, -21.1698856, 22.0648479
2: -5.4166827, 18.0737495, -5.0029421, 16.5449162, -21.9615993, 23.0766907
3: -6.5231509, 23.1954803, -5.9411888, 21.1416397, -27.6647911, 29.1366673
4: -5.3110218, 21.4500999, -4.8949690, 19.5565891, -24.8676109, 26.3450699

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4997424, upper bound: 27.5089839
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4997424, upper bound: 27.5204092
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.1293182, 14.6288443, -4.7206116, 16.2323303, -20.3616447, 19.3494530
1: -5.9055848, 14.9554243, -6.7455416, 16.6543579, -22.5599422, 21.7009659
2: -4.9716229, 16.8073635, -5.7037611, 18.6685505, -23.6401730, 22.5111217
3: -5.9947290, 21.5765266, -6.8226271, 23.8968391, -29.8915672, 28.3991547
4: -4.9104395, 19.9661102, -5.5644245, 22.1408634, -27.0513020, 25.5305328

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5046431, upper bound: 27.5097965
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5046431, upper bound: 27.5210142
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.5115433, 15.8061476, -4.7206116, 16.2323303, -20.7438698, 20.5267563
1: -6.4535074, 16.1446762, -6.7455416, 16.6543579, -23.1078644, 22.8902149
2: -5.4166827, 18.0737495, -5.7037611, 18.6685505, -24.0852337, 23.7775116
3: -6.5231509, 23.1954803, -6.8226271, 23.8968391, -30.4199905, 30.0181065
4: -5.3110218, 21.4500999, -5.5644245, 22.1408634, -27.4518852, 27.0145226

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5046431, upper bound: 27.5097965
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5046431, upper bound: 27.5210139
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.8377326, 13.7042408, -4.0815225, 14.3603363, -18.1980686, 17.7857609
1: -5.5586572, 14.0156584, -5.9201736, 14.7163811, -20.2750340, 19.9358330
2: -4.6897707, 15.7574348, -5.0029421, 16.5449162, -21.2346878, 20.7603760
3: -5.5885458, 20.1776047, -5.9411888, 21.1416397, -26.7301846, 26.1187916
4: -4.6097212, 18.6277122, -4.8949690, 19.5565891, -24.1663094, 23.5226822

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5233089, upper bound: 27.5233461
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5233089, upper bound: 27.5233461
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.1616135, 14.7958717, -4.0815225, 14.3603363, -18.5219498, 18.8773937
1: -6.0217419, 15.1112652, -5.9201736, 14.7163811, -20.7381191, 21.0314388
2: -5.0678825, 16.9106216, -5.0029421, 16.5449162, -21.6127987, 21.9135609
3: -6.0606642, 21.6794033, -5.9411888, 21.1416397, -27.2023048, 27.6205883
4: -4.9605665, 20.0012703, -4.8949690, 19.5565891, -24.5171547, 24.8962402

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5347714, upper bound: 27.5283614
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5347714, upper bound: 27.5283614
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.8377326, 13.7042408, -4.7206116, 16.2323303, -20.0700569, 18.4248486
1: -5.5586572, 14.0156584, -6.7455416, 16.6543579, -22.2130146, 20.7612000
2: -4.6897707, 15.7574348, -5.7037611, 18.6685505, -23.3583221, 21.4611931
3: -5.5885458, 20.1776047, -6.8226271, 23.8968391, -29.4853859, 27.0002308
4: -4.6097212, 18.6277122, -5.5644245, 22.1408634, -26.7505836, 24.1921368

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5233089, upper bound: 27.5239205
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5233089, upper bound: 27.5233461
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.1616135, 14.7958717, -4.7206116, 16.2323303, -20.3939362, 19.5164795
1: -6.0217419, 15.1112652, -6.7455416, 16.6543579, -22.6760998, 21.8568058
2: -5.0678825, 16.9106216, -5.7037611, 18.6685505, -23.7364330, 22.6143799
3: -6.0606642, 21.6794033, -6.8226271, 23.8968391, -29.9575043, 28.5020275
4: -4.9605665, 20.0012703, -5.5644245, 22.1408634, -27.1014290, 25.5656929

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5417291, upper bound: 27.5291740
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5417291, upper bound: 27.5403917
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.4364772, 15.4927216, -4.0815225, 14.3603363, -18.7968140, 19.5742435
1: -6.3414073, 15.8659678, -5.9201736, 14.7163811, -21.0577812, 21.7861404
2: -5.3552303, 17.7862968, -5.0029421, 16.5449162, -21.9001446, 22.7892380
3: -6.4260044, 22.8168812, -5.9411888, 21.1416397, -27.5676441, 28.7580700
4: -5.2513251, 21.1006165, -4.8949690, 19.5565891, -24.8079109, 25.9955864

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5241587, upper bound: 27.5303038
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5241587, upper bound: 27.5303038
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.7756276, 16.5842285, -4.0815225, 14.3603363, -19.1359634, 20.6657486
1: -6.8313746, 16.9627132, -5.9201736, 14.7163811, -21.5477524, 22.8828869
2: -5.7417583, 18.9490433, -5.0029421, 16.5449162, -22.2866707, 23.9519844
3: -6.9129910, 24.3177433, -5.9411888, 21.1416397, -28.0546265, 30.2589302
4: -5.6055784, 22.4766064, -4.8949690, 19.5565891, -25.1621628, 27.3715744

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5353764, upper bound: 27.5319601
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5353764, upper bound: 27.5433140
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.4364772, 15.4927216, -4.7206116, 16.2323303, -20.6688080, 20.2133293
1: -6.3414073, 15.8659678, -6.7455416, 16.6543579, -22.9957657, 22.6115093
2: -5.3552303, 17.7862968, -5.7037611, 18.6685505, -24.0237808, 23.4900551
3: -6.4260044, 22.8168812, -6.8226271, 23.8968391, -30.3228436, 29.6395073
4: -5.2513251, 21.1006165, -5.5644245, 22.1408634, -27.3921871, 26.6650391

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5306785, upper bound: 27.5311163
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5306785, upper bound: 27.5311163
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.7756276, 16.5842285, -4.7206116, 16.2323303, -21.0079536, 21.3048363
1: -6.8313746, 16.9627132, -6.7455416, 16.6543579, -23.4857330, 23.7082558
2: -5.7417583, 18.9490433, -5.7037611, 18.6685505, -24.4103088, 24.6528015
3: -6.9129910, 24.3177433, -6.8226271, 23.8968391, -30.8098259, 31.1403694
4: -5.6055784, 22.4766064, -5.5644245, 22.1408634, -27.7464390, 28.0410290

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5398042, upper bound: 27.5327727
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5398042, upper bound: 27.5439898
time: 0.81 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.27 seconds
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5064951, upper bound: 27.5072746
IS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5064951, upper bound: 27.5072746
IS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.4997424, upper bound: 27.5089837
IS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.4997424, upper bound: 27.5089839
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.4997424, upper bound: 27.5089839
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.4997424, upper bound: 27.5204092
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5046431, upper bound: 27.5097965
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5046431, upper bound: 27.5210142
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5046431, upper bound: 27.5097965
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5046431, upper bound: 27.5210139
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5233089, upper bound: 27.5233461
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5233089, upper bound: 27.5233461
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5347714, upper bound: 27.5283614
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5347714, upper bound: 27.5283614
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5233089, upper bound: 27.5239205
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5233089, upper bound: 27.5233461
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5417291, upper bound: 27.5291740
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5417291, upper bound: 27.5403917
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5241587, upper bound: 27.5303038
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5241587, upper bound: 27.5303038
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5353764, upper bound: 27.5319601
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5353764, upper bound: 27.5433140
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5306785, upper bound: 27.5311163
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5306785, upper bound: 27.5311163
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5398042, upper bound: 27.5327727
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -27.5398042, upper bound: 27.5439898

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.5115433, 15.8061476, -4.1616135, 14.7958717, -19.3074150, 19.9677582
1: -6.4535074, 16.1446762, -6.0217419, 15.1112652, -21.5647736, 22.1664143
2: -5.4166827, 18.0737495, -5.0678825, 16.9106216, -22.3273048, 23.1416321
3: -6.5231509, 23.1954803, -6.0606642, 21.6794033, -28.2025547, 29.2561455
4: -5.3110218, 21.4500999, -4.9605665, 20.0012703, -25.3122921, 26.4106674

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4925169, upper bound: 27.5046425
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4964113, upper bound: 27.5066369
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.1293182, 14.6288443, -4.7756276, 16.5842285, -20.7135468, 19.4044724
1: -5.9055848, 14.9554243, -6.8313746, 16.9627132, -22.8682976, 21.7867985
2: -4.9716229, 16.8073635, -5.7417583, 18.9490433, -23.9206619, 22.5491199
3: -5.9947290, 21.5765266, -6.9129910, 24.3177433, -30.3124733, 28.4895134
4: -4.9104395, 19.9661102, -5.6055784, 22.4766064, -27.3870449, 25.5716877

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5121156, upper bound: 27.5177676
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5158655, upper bound: 27.5206069
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.5115433, 15.8061476, -4.7756276, 16.5842285, -21.0957718, 20.5817738
1: -6.4535074, 16.1446762, -6.8313746, 16.9627132, -23.4162216, 22.9760494
2: -5.4166827, 18.0737495, -5.7417583, 18.9490433, -24.3657265, 23.8155079
3: -6.5231509, 23.1954803, -6.9129910, 24.3177433, -30.8408947, 30.1084671
4: -5.3110218, 21.4500999, -5.6055784, 22.4766064, -27.7876282, 27.0556755

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4941893, upper bound: 27.5054617
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5016666, upper bound: 27.5074979
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.8377326, 13.7042408, -3.8377326, 13.7042408, -17.5419731, 17.5419731
1: -5.5586572, 14.0156584, -5.5586572, 14.0156584, -19.5743160, 19.5743160
2: -4.6897707, 15.7574348, -4.6897707, 15.7574348, -20.4472046, 20.4472046
3: -5.5885458, 20.1776047, -5.5885458, 20.1776047, -25.7661514, 25.7661514
4: -4.6097212, 18.6277122, -4.6097212, 18.6277122, -23.2374344, 23.2374344

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5109197, upper bound: 27.4991147
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4896675, upper bound: 27.4911297
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.8377326, 13.7042408, -4.1616135, 14.7958717, -18.6336040, 17.8658524
1: -5.5586572, 14.0156584, -6.0217419, 15.1112652, -20.6699181, 20.0373993
2: -4.6897707, 15.7574348, -5.0678825, 16.9106216, -21.6003914, 20.8253174
3: -5.5885458, 20.1776047, -6.0606642, 21.6794033, -27.2679482, 26.2382679
4: -4.6097212, 18.6277122, -4.9605665, 20.0012703, -24.6109924, 23.5882797

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5109197, upper bound: 27.4991147
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4896675, upper bound: 27.4982271
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.1616135, 14.7958717, -3.8377326, 13.7042408, -17.8658524, 18.6336040
1: -6.0217419, 15.1112652, -5.5586572, 14.0156584, -20.0373993, 20.6699181
2: -5.0678825, 16.9106216, -4.6897707, 15.7574348, -20.8253174, 21.6003914
3: -6.0606642, 21.6794033, -5.5885458, 20.1776047, -26.2382660, 27.2679482
4: -4.9605665, 20.0012703, -4.6097212, 18.6277122, -23.5882797, 24.6109924

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5226030, upper bound: 27.5055987
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5212409, upper bound: 27.5046308
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.1616135, 14.7958717, -4.1616135, 14.7958717, -18.9574833, 18.9574833
1: -6.0217419, 15.1112652, -6.0217419, 15.1112652, -21.1330032, 21.1330032
2: -5.0678825, 16.9106216, -5.0678825, 16.9106216, -21.9785042, 21.9785042
3: -6.0606642, 21.6794033, -6.0606642, 21.6794033, -27.7400665, 27.7400665
4: -4.9605665, 20.0012703, -4.9605665, 20.0012703, -24.9618378, 24.9618378

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5226033, upper bound: 27.5055992
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5212413, upper bound: 27.5046310
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.8377326, 13.7042408, -4.4364772, 15.4927216, -19.3304539, 18.1407185
1: -5.5586572, 14.0156584, -6.3414073, 15.8659678, -21.4246254, 20.3570652
2: -4.6897707, 15.7574348, -5.3552303, 17.7862968, -22.4760666, 21.1126652
3: -5.5885458, 20.1776047, -6.4260044, 22.8168812, -28.4054260, 26.6036091
4: -4.6097212, 18.6277122, -5.2513251, 21.1006165, -25.7103386, 23.8790379

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5258424, upper bound: 27.5190219
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4945134, upper bound: 27.5042530
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.8377326, 13.7042408, -4.7756276, 16.5842285, -20.4219608, 18.4798679
1: -5.5586572, 14.0156584, -6.8313746, 16.9627132, -22.5213699, 20.8470325
2: -4.6897707, 15.7574348, -5.7417583, 18.9490433, -23.6388130, 21.4991913
3: -5.5885458, 20.1776047, -6.9129910, 24.3177433, -29.9062881, 27.0905895
4: -4.6097212, 18.6277122, -5.6055784, 22.4766064, -27.0863266, 24.2332897

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5258424, upper bound: 27.5190219
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4945134, upper bound: 27.5042530
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.1616135, 14.7958717, -4.4364772, 15.4927216, -19.6543350, 19.2323494
1: -6.0217419, 15.1112652, -6.3414073, 15.8659678, -21.8877106, 21.4526691
2: -5.0678825, 16.9106216, -5.3552303, 17.7862968, -22.8541794, 22.2658520
3: -6.0606642, 21.6794033, -6.4260044, 22.8168812, -28.8775444, 28.1054077
4: -4.9605665, 20.0012703, -5.2513251, 21.1006165, -26.0611839, 25.2525940

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5392073, upper bound: 27.5267365
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378452, upper bound: 27.5257680
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.1616135, 14.7958717, -4.7756276, 16.5842285, -20.7458382, 19.5714989
1: -6.0217419, 15.1112652, -6.8313746, 16.9627132, -22.9844551, 21.9426384
2: -5.0678825, 16.9106216, -5.7417583, 18.9490433, -24.0169258, 22.6523781
3: -6.0606642, 21.6794033, -6.9129910, 24.3177433, -30.3784065, 28.5923901
4: -4.9605665, 20.0012703, -5.6055784, 22.4766064, -27.4371719, 25.6068459

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5392076, upper bound: 27.5267365
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378457, upper bound: 27.5257683
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.4364772, 15.4927216, -3.8377326, 13.7042408, -18.1407185, 19.3304539
1: -6.3414073, 15.8659678, -5.5586572, 14.0156584, -20.3570652, 21.4246254
2: -5.3552303, 17.7862968, -4.6897707, 15.7574348, -21.1126652, 22.4760666
3: -6.4260044, 22.8168812, -5.5885458, 20.1776047, -26.6036091, 28.4054260
4: -5.2513251, 21.1006165, -4.6097212, 18.6277122, -23.8790379, 25.7103386

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5116007, upper bound: 27.5046927
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5122670, upper bound: 27.5077340
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.4364772, 15.4927216, -4.1616135, 14.7958717, -19.2323494, 19.6543331
1: -6.3414073, 15.8659678, -6.0217419, 15.1112652, -21.4526691, 21.8877106
2: -5.3552303, 17.7862968, -5.0678825, 16.9106216, -22.2658501, 22.8541794
3: -6.4260044, 22.8168812, -6.0606642, 21.6794033, -28.1054077, 28.8775444
4: -5.2513251, 21.1006165, -4.9605665, 20.0012703, -25.2525940, 26.0611839

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5116008, upper bound: 27.5046928
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5122670, upper bound: 27.5077340
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.7756276, 16.5842285, -3.8377326, 13.7042408, -18.4798660, 20.4219608
1: -6.8313746, 16.9627132, -5.5586572, 14.0156584, -20.8470325, 22.5213699
2: -5.7417583, 18.9490433, -4.6897707, 15.7574348, -21.4991932, 23.6388130
3: -6.9129910, 24.3177433, -5.5885458, 20.1776047, -27.0905895, 29.9062881
4: -5.6055784, 22.4766064, -4.6097212, 18.6277122, -24.2332897, 27.0863266

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5227553, upper bound: 27.5070346
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5236294, upper bound: 27.5096354
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.7756276, 16.5842285, -4.1616135, 14.7958717, -19.5714989, 20.7458382
1: -6.8313746, 16.9627132, -6.0217419, 15.1112652, -21.9426384, 22.9844551
2: -5.7417583, 18.9490433, -5.0678825, 16.9106216, -22.6523781, 24.0169258
3: -6.9129910, 24.3177433, -6.0606642, 21.6794033, -28.5923882, 30.3784065
4: -5.6055784, 22.4766064, -4.9605665, 20.0012703, -25.6068459, 27.4371719

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5227556, upper bound: 27.5070347
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5236297, upper bound: 27.5334078
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.4364772, 15.4927216, -4.4364772, 15.4927216, -19.9291992, 19.9291992
1: -6.3414073, 15.8659678, -6.3414073, 15.8659678, -22.2073746, 22.2073746
2: -5.3552303, 17.7862968, -5.3552303, 17.7862968, -23.1415253, 23.1415253
3: -6.4260044, 22.8168812, -6.4260044, 22.8168812, -29.2428856, 29.2428856
4: -5.2513251, 21.1006165, -5.2513251, 21.1006165, -26.3519402, 26.3519402

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5261766, upper bound: 27.5251796
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5287497, upper bound: 27.5288713
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.4364772, 15.4927216, -4.7756276, 16.5842285, -21.0207062, 20.2683487
1: -6.3414073, 15.8659678, -6.8313746, 16.9627132, -23.3041191, 22.6973419
2: -5.3552303, 17.7862968, -5.7417583, 18.9490433, -24.3042717, 23.5280514
3: -6.4260044, 22.8168812, -6.9129910, 24.3177433, -30.7437477, 29.7298679
4: -5.2513251, 21.1006165, -5.6055784, 22.4766064, -27.7279301, 26.7061920

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5261766, upper bound: 27.5251796
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5287497, upper bound: 27.5288713
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.7756276, 16.5842285, -4.4364772, 15.4927216, -20.2683487, 21.0207062
1: -6.8313746, 16.9627132, -6.3414073, 15.8659678, -22.6973419, 23.3041191
2: -5.7417583, 18.9490433, -5.3552303, 17.7862968, -23.5280533, 24.3042717
3: -6.9129910, 24.3177433, -6.4260044, 22.8168812, -29.7298698, 30.7437477
4: -5.6055784, 22.4766064, -5.2513251, 21.1006165, -26.7061920, 27.7279301

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5327023, upper bound: 27.5270950
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5380360, upper bound: 27.5307727
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.7756276, 16.5842285, -4.7756276, 16.5842285, -21.3598537, 21.3598537
1: -6.8313746, 16.9627132, -6.8313746, 16.9627132, -23.7940884, 23.7940884
2: -5.7417583, 18.9490433, -5.7417583, 18.9490433, -24.6907997, 24.6907978
3: -6.9129910, 24.3177433, -6.9129910, 24.3177433, -31.2307301, 31.2307301
4: -5.6055784, 22.4766064, -5.6055784, 22.4766064, -28.0821819, 28.0821819

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5327025, upper bound: 27.5270950
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5380361, upper bound: 27.5307727
time: 0.92 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.61 seconds
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.4925169, upper bound: 27.5046425
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.4964113, upper bound: 27.5066369
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5121156, upper bound: 27.5177676
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5158655, upper bound: 27.5206069
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.4941893, upper bound: 27.5054617
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5016666, upper bound: 27.5074979
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5109197, upper bound: 27.4991147
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.4896675, upper bound: 27.4911297
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5109197, upper bound: 27.4991147
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.4896675, upper bound: 27.4982271
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5226030, upper bound: 27.5055987
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5212409, upper bound: 27.5046308
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5226033, upper bound: 27.5055992
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5212413, upper bound: 27.5046310
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5258424, upper bound: 27.5190219
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.4945134, upper bound: 27.5042530
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5258424, upper bound: 27.5190219
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.4945134, upper bound: 27.5042530
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5392073, upper bound: 27.5267365
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5378452, upper bound: 27.5257680
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5392076, upper bound: 27.5267365
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5378457, upper bound: 27.5257683
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5116007, upper bound: 27.5046927
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5122670, upper bound: 27.5077340
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5116008, upper bound: 27.5046928
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5122670, upper bound: 27.5077340
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5227553, upper bound: 27.5070346
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5236294, upper bound: 27.5096354
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5227556, upper bound: 27.5070347
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5236297, upper bound: 27.5334078
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5261766, upper bound: 27.5251796
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5287497, upper bound: 27.5288713
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5261766, upper bound: 27.5251796
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5287497, upper bound: 27.5288713
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5327023, upper bound: 27.5270950
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5380360, upper bound: 27.5307727
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5327025, upper bound: 27.5270950
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.61
Output dim: 3, lower bound: -27.5380361, upper bound: 27.5307727

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.1791363, 14.7678900, -4.7756276, 16.5842285, -20.7633648, 19.5435181
1: -5.9261398, 15.0727625, -6.8313746, 16.9627132, -22.8888531, 21.9041367
2: -4.9837761, 16.9354610, -5.7417583, 18.9490433, -23.9328194, 22.6772156
3: -6.0110121, 21.7418270, -6.9129910, 24.3177433, -30.3287544, 28.6548119
4: -4.9157543, 20.1016560, -5.6055784, 22.4766064, -27.3923588, 25.7072315

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5158584, upper bound: 27.5311153
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5158584, upper bound: 27.5319894
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -3.8377326, 13.7042408, -17.3506985, 17.1879234
1: -5.2464833, 13.5860958, -5.5586572, 14.0156584, -19.2621422, 19.1447525
2: -4.3717928, 15.2154961, -4.6897707, 15.7574348, -20.1292267, 19.9052658
3: -5.2993393, 19.6015205, -5.5885458, 20.1776047, -25.4769440, 25.1900673
4: -4.3268204, 18.0051994, -4.6097212, 18.6277122, -22.9545326, 22.6149197

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5212412, upper bound: 27.5046309
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5212410, upper bound: 27.5046304
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.1191621, 14.6685381, -3.8377326, 13.7042408, -17.8234024, 18.5062714
1: -5.9215693, 14.9546156, -5.5586572, 14.0156584, -19.9372272, 20.5132732
2: -4.9671373, 16.7323151, -4.6897707, 15.7574348, -20.7245712, 21.4220848
3: -5.9684920, 21.4562817, -5.5885458, 20.1776047, -26.1460953, 27.0448265
4: -4.8678341, 19.7656593, -4.6097212, 18.6277122, -23.4955463, 24.3753815

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5212409, upper bound: 27.5046307
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5212409, upper bound: 27.5046307
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -4.1616135, 14.7958717, -18.4423294, 17.5118027
1: -5.2464833, 13.5860958, -6.0217419, 15.1112652, -20.3577461, 19.6078377
2: -4.3717928, 15.2154961, -5.0678825, 16.9106216, -21.2824135, 20.2833786
3: -5.2993393, 19.6015205, -6.0606642, 21.6794033, -26.9787407, 25.6621838
4: -4.3268204, 18.0051994, -4.9605665, 20.0012703, -24.3280907, 22.9657650

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5347420, upper bound: 27.5328198
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5347420, upper bound: 27.5328198
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.1191621, 14.6685381, -4.1616135, 14.7958717, -18.9150333, 18.8301506
1: -5.9215693, 14.9546156, -6.0217419, 15.1112652, -21.0328293, 20.9763565
2: -4.9671373, 16.7323151, -5.0678825, 16.9106216, -21.8777580, 21.8001976
3: -5.9684920, 21.4562817, -6.0606642, 21.6794033, -27.6478920, 27.5169430
4: -4.8678341, 19.7656593, -4.9605665, 20.0012703, -24.8691044, 24.7262268

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5347419, upper bound: 27.5328198
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5347419, upper bound: 27.5328193
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.3112977, 12.2320118, -4.4364772, 15.4927216, -18.8040199, 16.6684875
1: -4.7633290, 12.4608803, -6.3414073, 15.8659678, -20.6292973, 18.8022881
2: -3.9792581, 14.0205498, -5.3552303, 17.7862968, -21.7655525, 19.3757782
3: -4.8087497, 18.0676193, -6.4260044, 22.8168812, -27.6256313, 24.4936237
4: -3.9650612, 16.5860634, -5.2513251, 21.1006165, -25.0656776, 21.8373890

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4994109, upper bound: 27.5047237
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4994109, upper bound: 27.5057743
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.3112977, 12.2320118, -4.7756276, 16.5842285, -19.8955269, 17.0076370
1: -4.7633290, 12.4608803, -6.8313746, 16.9627132, -21.7260418, 19.2922554
2: -3.9792581, 14.0205498, -5.7417583, 18.9490433, -22.9282990, 19.7623062
3: -4.8087497, 18.0676193, -6.9129910, 24.3177433, -29.1264935, 24.9806061
4: -3.9650612, 16.5860634, -5.6055784, 22.4766064, -26.4416676, 22.1916428

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4897051, upper bound: 27.5008154
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4897053, upper bound: 27.5008157
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -4.4364772, 15.4927216, -19.1391792, 17.7866688
1: -5.2464833, 13.5860958, -6.3414073, 15.8659678, -21.1124516, 19.9275036
2: -4.3717928, 15.2154961, -5.3552303, 17.7862968, -22.1580887, 20.5707245
3: -5.2993393, 19.6015205, -6.4260044, 22.8168812, -28.1162205, 26.0275249
4: -4.3268204, 18.0051994, -5.2513251, 21.1006165, -25.4274368, 23.2565212

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5348042, upper bound: 27.5251019
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5348042, upper bound: 27.5257682
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.1191621, 14.6685381, -4.4364772, 15.4927216, -19.6118832, 19.1050148
1: -5.9215693, 14.9546156, -6.3414073, 15.8659678, -21.7875366, 21.2960224
2: -4.9671373, 16.7323151, -5.3552303, 17.7862968, -22.7534332, 22.0875454
3: -5.9684920, 21.4562817, -6.4260044, 22.8168812, -28.7853737, 27.8822861
4: -4.8678341, 19.7656593, -5.2513251, 21.1006165, -25.9684505, 25.0169830

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5348039, upper bound: 27.5251018
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5348039, upper bound: 27.5257680
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -4.7756276, 16.5842285, -20.2306862, 18.1258183
1: -5.2464833, 13.5860958, -6.8313746, 16.9627132, -22.2091942, 20.4174709
2: -4.3717928, 15.2154961, -5.7417583, 18.9490433, -23.3208351, 20.9572525
3: -5.2993393, 19.6015205, -6.9129910, 24.3177433, -29.6170826, 26.5145073
4: -4.3268204, 18.0051994, -5.6055784, 22.4766064, -26.8034267, 23.6107731

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5368589, upper bound: 27.5345978
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5368589, upper bound: 27.5364979
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.1191621, 14.6685381, -4.7756276, 16.5842285, -20.7033901, 19.4441662
1: -5.9215693, 14.9546156, -6.8313746, 16.9627132, -22.8842812, 21.7859898
2: -4.9671373, 16.7323151, -5.7417583, 18.9490433, -23.9161797, 22.4740715
3: -5.9684920, 21.4562817, -6.9129910, 24.3177433, -30.2862320, 28.3692665
4: -4.8678341, 19.7656593, -5.6055784, 22.4766064, -27.3444405, 25.3712349

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5368590, upper bound: 27.5345981
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5368590, upper bound: 27.5364980
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.2396750, 15.1407347, -3.8377326, 13.7042408, -17.9439144, 18.9784660
1: -6.0757089, 15.4544592, -5.5586572, 14.0156584, -20.0913677, 21.0131130
2: -5.0678725, 17.2726898, -4.6897707, 15.7574348, -20.8253078, 21.9624596
3: -6.1715178, 22.2449474, -5.5885458, 20.1776047, -26.3491230, 27.8334923
4: -5.0068088, 20.4940319, -4.6097212, 18.6277122, -23.6345215, 25.1037521

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5227555, upper bound: 27.5070346
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5227555, upper bound: 27.5070346
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.8151884, 16.7127552, -3.8377326, 13.7042408, -18.5194244, 20.5504875
1: -6.8487349, 17.0677147, -5.5586572, 14.0156584, -20.8643932, 22.6263695
2: -5.7475977, 19.0672207, -4.6897707, 15.7574348, -21.5050316, 23.7569904
3: -6.9301233, 24.4769001, -5.5885458, 20.1776047, -27.1077271, 30.0654449
4: -5.6081338, 22.6030693, -4.6097212, 18.6277122, -24.2358456, 27.2127914

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5236297, upper bound: 27.5096354
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5236297, upper bound: 27.5096354
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.2396750, 15.1407347, -4.1616135, 14.7958717, -19.0355473, 19.3023453
1: -6.0757089, 15.4544592, -6.0217419, 15.1112652, -21.1869678, 21.4761982
2: -5.0678725, 17.2726898, -5.0678825, 16.9106216, -21.9784946, 22.3405724
3: -6.1715178, 22.2449474, -6.0606642, 21.6794033, -27.8509216, 28.3056107
4: -5.0068088, 20.4940319, -4.9605665, 20.0012703, -25.0080795, 25.4545975

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5362564, upper bound: 27.5329350
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5362564, upper bound: 27.5329358
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.8151884, 16.7127552, -4.1616135, 14.7958717, -19.6110573, 20.8743687
1: -6.8487349, 17.0677147, -6.0217419, 15.1112652, -21.9599991, 23.0894547
2: -5.7475977, 19.0672207, -5.0678825, 16.9106216, -22.6582165, 24.1351032
3: -6.9301233, 24.4769001, -6.0606642, 21.6794033, -28.6095276, 30.5375633
4: -5.6081338, 22.6030693, -4.9605665, 20.0012703, -25.6094036, 27.5636349

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5371304, upper bound: 27.5334066
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5371304, upper bound: 27.5334066
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.8882565, 14.0178547, -4.4364772, 15.4927216, -19.3809776, 18.4543324
1: -5.5631576, 14.3240805, -6.3414073, 15.8659678, -21.4291248, 20.6654854
2: -4.6534100, 16.0694389, -5.3552303, 17.7862968, -22.4397068, 21.4246655
3: -5.6674542, 20.6948471, -6.4260044, 22.8168812, -28.4843349, 27.1208515
4: -4.6273518, 19.0684605, -5.2513251, 21.1006165, -25.7279682, 24.3197842

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5249858, upper bound: 27.5251523
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5249858, upper bound: 27.5251796
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.4765372, 15.6225147, -4.4364772, 15.4927216, -19.9692574, 20.0589924
1: -6.3536415, 15.9724932, -6.3414073, 15.8659678, -22.2196083, 22.3139000
2: -5.3602648, 17.9039478, -5.3552303, 17.7862968, -23.1465607, 23.2591763
3: -6.4377475, 22.9774036, -6.4260044, 22.8168812, -29.2546291, 29.4034081
4: -5.2538066, 21.2279301, -5.2513251, 21.1006165, -26.3544216, 26.4792538

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5258300, upper bound: 27.5282051
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5258300, upper bound: 27.5288713
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.8882565, 14.0178547, -4.7756276, 16.5842285, -20.4724808, 18.7934818
1: -5.5631576, 14.3240805, -6.8313746, 16.9627132, -22.5258694, 21.1554546
2: -4.6534100, 16.0694389, -5.7417583, 18.9490433, -23.6024532, 21.8111935
3: -5.6674542, 20.6948471, -6.9129910, 24.3177433, -29.9851971, 27.6078339
4: -4.6273518, 19.0684605, -5.6055784, 22.4766064, -27.1039581, 24.6740360

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5260153, upper bound: 27.5334170
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5260152, upper bound: 27.5334167
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.4765372, 15.6225147, -4.7756276, 16.5842285, -21.0607624, 20.3981400
1: -6.3536415, 15.9724932, -6.8313746, 16.9627132, -23.3163548, 22.8038673
2: -5.3602648, 17.9039478, -5.7417583, 18.9490433, -24.3093071, 23.6457043
3: -6.4377475, 22.9774036, -6.9129910, 24.3177433, -30.7554913, 29.8903904
4: -5.2538066, 21.2279301, -5.6055784, 22.4766064, -27.7304115, 26.8335056

Time for backsubstitution: 2.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5275652, upper bound: 27.5378474
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5275651, upper bound: 27.5394851
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.2396750, 15.1407347, -4.4364772, 15.4927216, -19.7323952, 19.5772114
1: -6.0757089, 15.4544592, -6.3414073, 15.8659678, -21.9416771, 21.7958641
2: -5.0678725, 17.2726898, -5.3552303, 17.7862968, -22.8541679, 22.6279202
3: -6.1715178, 22.2449474, -6.4260044, 22.8168812, -28.9883995, 28.6709518
4: -5.0068088, 20.4940319, -5.2513251, 21.1006165, -26.1074257, 25.7453556

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5327024, upper bound: 27.5270950
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5327024, upper bound: 27.5270950
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.8151884, 16.7127552, -4.4364772, 15.4927216, -20.3079071, 21.1492329
1: -6.8487349, 17.0677147, -6.3414073, 15.8659678, -22.7147026, 23.4091225
2: -5.7475977, 19.0672207, -5.3552303, 17.7862968, -23.5338898, 24.4224491
3: -6.9301233, 24.4769001, -6.4260044, 22.8168812, -29.7470055, 30.9029045
4: -5.6081338, 22.6030693, -5.2513251, 21.1006165, -26.7087498, 27.8543911

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5371680, upper bound: 27.5301065
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5371678, upper bound: 27.5307727
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.2396750, 15.1407347, -4.7756276, 16.5842285, -20.8239002, 19.9163609
1: -6.0757089, 15.4544592, -6.8313746, 16.9627132, -23.0384197, 22.2858334
2: -5.0678725, 17.2726898, -5.7417583, 18.9490433, -24.0169144, 23.0144482
3: -6.1715178, 22.2449474, -6.9129910, 24.3177433, -30.4892616, 29.1579342
4: -5.0068088, 20.4940319, -5.6055784, 22.4766064, -27.4834156, 26.0996094

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5379823, upper bound: 27.5368323
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5379823, upper bound: 27.5369100
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.8151884, 16.7127552, -4.7756276, 16.5842285, -21.3994122, 21.4883823
1: -6.8487349, 17.0677147, -6.8313746, 16.9627132, -23.8114471, 23.8990898
2: -5.7475977, 19.0672207, -5.7417583, 18.9490433, -24.6966362, 24.8089752
3: -6.9301233, 24.4769001, -6.9129910, 24.3177433, -31.2478676, 31.3898869
4: -5.6081338, 22.6030693, -5.6055784, 22.4766064, -28.0847397, 28.2086430

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5392167, upper bound: 27.5377040
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5392168, upper bound: 27.5388739
time: 0.86 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.54 seconds
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5158584, upper bound: 27.5311153
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5158584, upper bound: 27.5319894
IS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5212412, upper bound: 27.5046309
IS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5212410, upper bound: 27.5046304
IS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5212409, upper bound: 27.5046307
IS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5212409, upper bound: 27.5046307
IS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5347420, upper bound: 27.5328198
IS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5347420, upper bound: 27.5328198
IS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5347419, upper bound: 27.5328198
IS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5347419, upper bound: 27.5328193
IS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.4994109, upper bound: 27.5047237
IS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.4994109, upper bound: 27.5057743
IS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.4897051, upper bound: 27.5008154
IS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.4897053, upper bound: 27.5008157
IS_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5348042, upper bound: 27.5251019
IS_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5348042, upper bound: 27.5257682
IS_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5348039, upper bound: 27.5251018
IS_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5348039, upper bound: 27.5257680
IS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5368589, upper bound: 27.5345978
IS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5368589, upper bound: 27.5364979
IS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5368590, upper bound: 27.5345981
IS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5368590, upper bound: 27.5364980
IS_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5227555, upper bound: 27.5070346
IS_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5227555, upper bound: 27.5070346
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5236297, upper bound: 27.5096354
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5236297, upper bound: 27.5096354
IS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5362564, upper bound: 27.5329350
IS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5362564, upper bound: 27.5329358
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5371304, upper bound: 27.5334066
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5371304, upper bound: 27.5334066
IS_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5249858, upper bound: 27.5251523
IS_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5249858, upper bound: 27.5251796
IS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5258300, upper bound: 27.5282051
IS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5258300, upper bound: 27.5288713
IS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5260153, upper bound: 27.5334170
IS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5260152, upper bound: 27.5334167
IS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5275652, upper bound: 27.5378474
IS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5275651, upper bound: 27.5394851
IS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5327024, upper bound: 27.5270950
IS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5327024, upper bound: 27.5270950
IS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5371680, upper bound: 27.5301065
IS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5371678, upper bound: 27.5307727
IS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5379823, upper bound: 27.5368323
IS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5379823, upper bound: 27.5369100
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5392167, upper bound: 27.5377040
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.54
Output dim: 3, lower bound: -27.5392168, upper bound: 27.5388739

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.1791363, 14.7678900, -4.2396750, 15.1407347, -19.3198700, 19.0075645
1: -5.9261398, 15.0727625, -6.0757089, 15.4544592, -21.3805981, 21.1484661
2: -4.9837761, 16.9354610, -5.0678725, 17.2726898, -22.2564659, 22.0033321
3: -6.0110121, 21.7418270, -6.1715178, 22.2449474, -28.2559586, 27.9133453
4: -4.9157543, 20.1016560, -5.0068088, 20.4940319, -25.4097824, 25.1084652

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5070813, upper bound: 27.5307262
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5156949, upper bound: 27.5307434
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.1791363, 14.7678900, -4.8151884, 16.7127552, -20.8918915, 19.5830765
1: -5.9261398, 15.0727625, -6.8487349, 17.0677147, -22.9938545, 21.9214954
2: -4.9837761, 16.9354610, -5.7475977, 19.0672207, -24.0509968, 22.6830559
3: -6.0110121, 21.7418270, -6.9301233, 24.4769001, -30.4879112, 28.6719513
4: -4.9157543, 20.1016560, -5.6081338, 22.6030693, -27.5188217, 25.7097893

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5070813, upper bound: 27.5316483
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5156949, upper bound: 27.5316654
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -3.3112977, 12.2320118, -15.8784695, 16.6614895
1: -5.2464833, 13.5860958, -4.7633290, 12.4608803, -17.7073612, 18.3494244
2: -4.3717928, 15.2154961, -3.9792581, 14.0205498, -18.3923416, 19.1947536
3: -5.2993393, 19.6015205, -4.8087497, 18.0676193, -23.3669586, 24.4102707
4: -4.3268204, 18.0051994, -3.9650612, 16.5860634, -20.9128838, 21.9702606

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5087122, upper bound: 27.5053748
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5217726, upper bound: 27.5055992
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -3.8090727, 13.6119699, -17.2584229, 17.1592655
1: -5.2464833, 13.5860958, -5.4754810, 13.8948555, -19.1413383, 19.0615768
2: -4.3717928, 15.2154961, -4.6048141, 15.6138811, -19.9856720, 19.8203106
3: -5.2993393, 19.6015205, -5.5135355, 20.0045624, -25.3039017, 25.1150551
4: -4.3268204, 18.0051994, -4.5309124, 18.4355297, -22.7623501, 22.5361118

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5087122, upper bound: 27.5053748
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5217728, upper bound: 27.5055992
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.1191621, 14.6685381, -3.3112977, 12.2320118, -16.3511734, 17.9798355
1: -5.9215693, 14.9546156, -4.7633290, 12.4608803, -18.3824482, 19.7179451
2: -4.9671373, 16.7323151, -3.9792581, 14.0205498, -18.9876862, 20.7115726
3: -5.9684920, 21.4562817, -4.8087497, 18.0676193, -24.0361099, 26.2650318
4: -4.8678341, 19.7656593, -3.9650612, 16.5860634, -21.4538975, 23.7307205

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5092667, upper bound: 27.5046308
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5203525, upper bound: 27.5043496
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.1191621, 14.6685381, -3.8090727, 13.6119699, -17.7311325, 18.4776115
1: -5.9215693, 14.9546156, -5.4754810, 13.8948555, -19.8164253, 20.4300957
2: -4.9671373, 16.7323151, -4.6048141, 15.6138811, -20.5810165, 21.3371296
3: -5.9684920, 21.4562817, -5.5135355, 20.0045624, -25.9730511, 26.9698162
4: -4.8678341, 19.7656593, -4.5309124, 18.4355297, -23.3033638, 24.2965717

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5092672, upper bound: 27.5046309
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5203528, upper bound: 27.5043498
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -3.6464579, 13.3501921, -16.9966488, 16.9966488
1: -5.2464833, 13.5860958, -5.2464833, 13.5860958, -18.8325787, 18.8325787
2: -4.3717928, 15.2154961, -4.3717928, 15.2154961, -19.5872879, 19.5872879
3: -5.2993393, 19.6015205, -5.2993393, 19.6015205, -24.9008598, 24.9008598
4: -4.3268204, 18.0051994, -4.3268204, 18.0051994, -22.3320179, 22.3320179

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5224149, upper bound: 27.5324873
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5354754, upper bound: 27.5324614
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -4.1191621, 14.6685381, -18.3149967, 17.4693546
1: -5.2464833, 13.5860958, -5.9215693, 14.9546156, -20.2010994, 19.5076656
2: -4.3717928, 15.2154961, -4.9671373, 16.7323151, -21.1041069, 20.1826324
3: -5.2993393, 19.6015205, -5.9684920, 21.4562817, -26.7556190, 25.5700092
4: -4.3268204, 18.0051994, -4.8678341, 19.7656593, -24.0924797, 22.8730335

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5224149, upper bound: 27.5324873
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5354754, upper bound: 27.5324614
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.1191621, 14.6685381, -3.6464579, 13.3501921, -17.4693546, 18.3149967
1: -5.9215693, 14.9546156, -5.2464833, 13.5860958, -19.5076656, 20.2010994
2: -4.9671373, 16.7323151, -4.3717928, 15.2154961, -20.1826324, 21.1041069
3: -5.9684920, 21.4562817, -5.2993393, 19.6015205, -25.5700092, 26.7556210
4: -4.8678341, 19.7656593, -4.3268204, 18.0051994, -22.8730335, 24.0924797

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5224580, upper bound: 27.5323993
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5340554, upper bound: 27.5323893
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.1191621, 14.6685381, -4.1191621, 14.6685381, -18.7877007, 18.7877007
1: -5.9215693, 14.9546156, -5.9215693, 14.9546156, -20.8761845, 20.8761845
2: -4.9671373, 16.7323151, -4.9671373, 16.7323151, -21.6994514, 21.6994514
3: -5.9684920, 21.4562817, -5.9684920, 21.4562817, -27.4247704, 27.4247723
4: -4.8678341, 19.7656593, -4.8678341, 19.7656593, -24.6334934, 24.6334934

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5224580, upper bound: 27.5323993
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5340553, upper bound: 27.5323893
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -3.8882565, 14.0178547, -17.6643124, 17.2384453
1: -5.2464833, 13.5860958, -5.5631576, 14.3240805, -19.5705624, 19.1492538
2: -4.3717928, 15.2154961, -4.6534100, 16.0694389, -20.4412289, 19.8689060
3: -5.2993393, 19.6015205, -5.6674542, 20.6948471, -25.9941864, 25.2689743
4: -4.3268204, 18.0051994, -4.6273518, 19.0684605, -23.3952808, 22.6325512

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5067265, upper bound: 27.5082477
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4924927, upper bound: 27.4758690
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -4.4765372, 15.6225147, -19.2689705, 17.8267288
1: -5.2464833, 13.5860958, -6.3536415, 15.9724932, -21.2189770, 19.9397373
2: -4.3717928, 15.2154961, -5.3602648, 17.9039478, -22.2757397, 20.5757599
3: -5.2993393, 19.6015205, -6.4377475, 22.9774036, -28.2767410, 26.0392685
4: -4.3268204, 18.0051994, -5.2538066, 21.2279301, -25.5547504, 23.2590027

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5067275, upper bound: 27.5165058
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4924928, upper bound: 27.4799055
time: 1.04 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 4.83 seconds
IS_A1_B2_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5070813, upper bound: 27.5307262
IS_A1_B2_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5156949, upper bound: 27.5307434
IS_A1_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5070813, upper bound: 27.5316483
IS_A1_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5156949, upper bound: 27.5316654
IS_A2_B2_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5087122, upper bound: 27.5053748
IS_A2_B2_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5217726, upper bound: 27.5055992
IS_A2_B2_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5087122, upper bound: 27.5053748
IS_A2_B2_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5217728, upper bound: 27.5055992
IS_A2_B2_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5092667, upper bound: 27.5046308
IS_A2_B2_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5203525, upper bound: 27.5043496
IS_A2_B2_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5092672, upper bound: 27.5046309
IS_A2_B2_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5203528, upper bound: 27.5043498
IS_A2_B2_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5224149, upper bound: 27.5324873
IS_A2_B2_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5354754, upper bound: 27.5324614
IS_A2_B2_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5224149, upper bound: 27.5324873
IS_A2_B2_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5354754, upper bound: 27.5324614
IS_A2_B2_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5224580, upper bound: 27.5323993
IS_A2_B2_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5340554, upper bound: 27.5323893
IS_A2_B2_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5224580, upper bound: 27.5323993
IS_A2_B2_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5340553, upper bound: 27.5323893
IS_A2_B2_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5067265, upper bound: 27.5082477
IS_A2_B2_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.4924927, upper bound: 27.4758690
IS_A2_B2_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.5067275, upper bound: 27.5165058
IS_A2_B2_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.83
Output dim: 3, lower bound: -27.4924928, upper bound: 27.4799055
IS_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5348039, upper bound: 27.5251018
IS_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5348039, upper bound: 27.5257680
IS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5368589, upper bound: 27.5345978
IS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5368589, upper bound: 27.5364979
IS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5368590, upper bound: 27.5345981
IS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5368590, upper bound: 27.5364980
IS_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5227555, upper bound: 27.5070346
IS_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5227555, upper bound: 27.5070346
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5236297, upper bound: 27.5096354
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5236297, upper bound: 27.5096354
IS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5362564, upper bound: 27.5329350
IS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5362564, upper bound: 27.5329358
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5371304, upper bound: 27.5334066
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5371304, upper bound: 27.5334066
IS_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5249858, upper bound: 27.5251523
IS_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5249858, upper bound: 27.5251796
IS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5258300, upper bound: 27.5282051
IS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5258300, upper bound: 27.5288713
IS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5260153, upper bound: 27.5334170
IS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5260152, upper bound: 27.5334167
IS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5275652, upper bound: 27.5378474
IS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5275651, upper bound: 27.5394851
IS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5327024, upper bound: 27.5270950
IS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5327024, upper bound: 27.5270950
IS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5371680, upper bound: 27.5301065
IS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5371678, upper bound: 27.5307727
IS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5379823, upper bound: 27.5368323
IS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5379823, upper bound: 27.5369100
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5392167, upper bound: 27.5377040
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -27.5392168, upper bound: 27.5388739
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=31.572500228881836
rel_dist={3: [-27.547796646245764, 27.54779664624577]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5274061, upper bound: 27.5365351
time: 0.79 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5434465, upper bound: 27.5434466
time: 0.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.83 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.83
Output dim: 3, lower bound: -27.5274061, upper bound: 27.5365351
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.83
Output dim: 3, lower bound: -27.5434465, upper bound: 27.5434466

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.4779983, 15.6032467, -4.8926516, 16.6545238, -21.1325207, 20.4958973
1: -6.4072104, 15.9771137, -6.9941020, 17.1100998, -23.5173073, 22.9712143
2: -5.4055552, 17.9303551, -5.9263554, 19.1902447, -24.5958004, 23.8567104
3: -6.4871049, 22.9893456, -7.0618343, 24.5106659, -30.9977703, 30.0511799
4: -5.3025470, 21.2900162, -5.7611084, 22.7441978, -28.0467453, 27.0511246

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5204525, upper bound: 27.5204525
time: 0.83 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5204525, upper bound: 27.5365351
time: 0.71 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -4.8002801, 16.4875641, -4.8926516, 16.6545238, -21.4548035, 21.3802128
1: -6.8583674, 16.9091358, -6.9941020, 17.1100998, -23.9684658, 23.9032364
2: -5.7989798, 18.9445438, -5.9263554, 19.1902447, -24.9892235, 24.8708992
3: -6.9374380, 24.2584801, -7.0618343, 24.5106659, -31.4481030, 31.3203144
4: -5.6505985, 22.4603367, -5.7611084, 22.7441978, -28.3947964, 28.2214451

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5365351, upper bound: 27.5274061
time: 0.70 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5365351, upper bound: 27.5434466
time: 0.84 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.08 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 3, lower bound: -27.5204525, upper bound: 27.5204525
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 3, lower bound: -27.5204525, upper bound: 27.5365351
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 3, lower bound: -27.5365351, upper bound: 27.5274061
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 3, lower bound: -27.5365351, upper bound: 27.5434466

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -4.4779983, 15.6032467, -4.4779983, 15.6032467, -20.0812454, 20.0812454
1: -6.4072104, 15.9771137, -6.4072104, 15.9771137, -22.3843231, 22.3843231
2: -5.4055552, 17.9303551, -5.4055552, 17.9303551, -23.3359108, 23.3359108
3: -6.4871049, 22.9893456, -6.4871049, 22.9893456, -29.4764500, 29.4764500
4: -5.3025470, 21.2900162, -5.3025470, 21.2900162, -26.5925636, 26.5925636

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5194443, upper bound: 27.5174105
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5194443, upper bound: 27.5174105
time: 0.62 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -4.4779983, 15.6032467, -4.8002801, 16.4875641, -20.9655628, 20.4035263
1: -6.4072104, 15.9771137, -6.8583674, 16.9091358, -23.3163452, 22.8354797
2: -5.4055552, 17.9303551, -5.7989798, 18.9445438, -24.3500996, 23.7293358
3: -6.4871049, 22.9893456, -6.9374380, 24.2584801, -30.7455845, 29.9267845
4: -5.3025470, 21.2900162, -5.6505985, 22.4603367, -27.7628841, 26.9406128

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5194443, upper bound: 27.5333702
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5194443, upper bound: 27.5174105
time: 0.86 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -4.8002801, 16.4875641, -4.4779983, 15.6032467, -20.4035263, 20.9655628
1: -6.8583674, 16.9091358, -6.4072104, 15.9771137, -22.8354816, 23.3163452
2: -5.7989798, 18.9445438, -5.4055552, 17.9303551, -23.7293358, 24.3500996
3: -6.9374380, 24.2584801, -6.4871049, 22.9893456, -29.9267845, 30.7455845
4: -5.6505985, 22.4603367, -5.3025470, 21.2900162, -26.9406128, 27.7628841

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5355028, upper bound: 27.5237309
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5204525, upper bound: 27.5204525
time: 0.68 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -4.8002801, 16.4875641, -4.8002801, 16.4875641, -21.2878418, 21.2878418
1: -6.8583674, 16.9091358, -6.8583674, 16.9091358, -23.7675037, 23.7675018
2: -5.7989798, 18.9445438, -5.7989798, 18.9445438, -24.7435226, 24.7435226
3: -6.9374380, 24.2584801, -6.9374380, 24.2584801, -31.1959171, 31.1959171
4: -5.6505985, 22.4603367, -5.6505985, 22.4603367, -28.1109352, 28.1109352

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5355029, upper bound: 27.5397497
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5365351, upper bound: 27.5433274
time: 0.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.20 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.20
Output dim: 3, lower bound: -27.5194443, upper bound: 27.5174105
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.20
Output dim: 3, lower bound: -27.5194443, upper bound: 27.5174105
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 3, lower bound: -27.5194443, upper bound: 27.5333702
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.20
Output dim: 3, lower bound: -27.5194443, upper bound: 27.5174105
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 3, lower bound: -27.5355028, upper bound: 27.5237309
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 3, lower bound: -27.5204525, upper bound: 27.5204525
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 3, lower bound: -27.5355029, upper bound: 27.5397497
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 3, lower bound: -27.5365351, upper bound: 27.5433274

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.8096104, 13.5872917, -4.6171923, 15.9559765, -19.7655869, 18.2044830
1: -5.5293055, 13.9026499, -6.5899792, 16.3524437, -21.8817482, 20.4926300
2: -4.6612096, 15.6515274, -5.5695691, 18.3252964, -22.9865055, 21.2210960
3: -5.5544548, 20.0211716, -6.6678381, 23.4805641, -29.0350189, 26.6890106
4: -4.5881572, 18.5381012, -5.4431725, 21.7361221, -26.3242779, 23.9812737

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5163087, upper bound: 27.5323992
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5163087, upper bound: 27.5333701
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.0815225, 14.3603363, -4.3009157, 15.0806694, -19.1621914, 18.6612511
1: -5.9201736, 14.7163811, -6.1504359, 15.4315176, -21.3516922, 20.8668118
2: -5.0029421, 16.5449162, -5.1821733, 17.3254128, -22.3283520, 21.7270889
3: -5.9411888, 21.1416397, -6.2218862, 22.2226276, -28.1638145, 27.3635235
4: -4.8949690, 19.5565891, -5.0986385, 20.5765514, -25.4715195, 24.6552277

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5323992, upper bound: 27.5226259
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5323992, upper bound: 27.5226259
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.7206116, 16.2323303, -4.4717021, 15.5833225, -20.3039303, 20.7040310
1: -6.7455416, 16.6543579, -6.3983030, 15.9571428, -22.7026825, 23.0526619
2: -5.7037611, 18.6685505, -5.3979626, 17.9085236, -23.6122818, 24.0665131
3: -6.8226271, 23.8968391, -6.4780731, 22.9608898, -29.7835159, 30.3749123
4: -5.5644245, 22.1408634, -5.2956080, 21.2647552, -26.8291798, 27.4364700

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5333701, upper bound: 27.5260141
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5333701, upper bound: 27.5272412
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.0815225, 14.3603363, -4.6171923, 15.9559765, -20.0374966, 18.9775276
1: -5.9201736, 14.7163811, -6.5899792, 16.3524437, -22.2726173, 21.3063583
2: -5.0029421, 16.5449162, -5.5695691, 18.3252964, -23.3282356, 22.1144848
3: -5.9411888, 21.1416397, -6.6678381, 23.4805641, -29.4217510, 27.8094749
4: -4.8949690, 19.5565891, -5.4431725, 21.7361221, -26.6310921, 24.9997616

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387120, upper bound: 27.5387279
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387120, upper bound: 27.5397495
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.7206116, 16.2323303, -4.7941203, 16.4680023, -21.1886120, 21.0264473
1: -6.7455416, 16.6543579, -6.8495827, 16.8895111, -23.6350498, 23.5039406
2: -5.7037611, 18.6685505, -5.7915468, 18.9231434, -24.6269016, 24.4600983
3: -6.8226271, 23.8968391, -6.9285007, 24.2305489, -31.0531731, 30.8253345
4: -5.5644245, 22.1408634, -5.6437979, 22.4355240, -27.9999466, 27.7846584

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5397153, upper bound: 27.5420219
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5397153, upper bound: 27.5433272
time: 0.65 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.19 seconds
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5163087, upper bound: 27.5323992
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5163087, upper bound: 27.5333701
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5323992, upper bound: 27.5226259
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5323992, upper bound: 27.5226259
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5333701, upper bound: 27.5260141
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5333701, upper bound: 27.5272412
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5387120, upper bound: 27.5387279
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5387120, upper bound: 27.5397495
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5397153, upper bound: 27.5420219
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5397153, upper bound: 27.5433272

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.8096104, 13.5872917, -4.0815225, 14.3603363, -18.1699467, 17.6688137
1: -5.5293055, 13.9026499, -5.9201736, 14.7163811, -20.2456837, 19.8228226
2: -4.6612096, 15.6515274, -5.0029421, 16.5449162, -21.2061253, 20.6544666
3: -5.5544548, 20.0211716, -5.9411888, 21.1416397, -26.6960945, 25.9623604
4: -4.5881572, 18.5381012, -4.8949690, 19.5565891, -24.1447430, 23.4330711

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5120794, upper bound: 27.5263943
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5021641, upper bound: 27.5177531
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.8096104, 13.5872917, -4.7206116, 16.2323303, -20.0419388, 18.3079014
1: -5.5293055, 13.9026499, -6.7455416, 16.6543579, -22.1836624, 20.6481915
2: -4.6612096, 15.6515274, -5.7037611, 18.6685505, -23.3297596, 21.3552837
3: -5.5544548, 20.0211716, -6.8226271, 23.8968391, -29.4512939, 26.8437996
4: -4.5881572, 18.5381012, -5.5644245, 22.1408634, -26.7290192, 24.1025238

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5120794, upper bound: 27.5269993
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5021641, upper bound: 27.5183984
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.0815225, 14.3603363, -3.8096104, 13.5872917, -17.6688137, 18.1699467
1: -5.9201736, 14.7163811, -5.5293055, 13.9026499, -19.8228226, 20.2456818
2: -5.0029421, 16.5449162, -4.6612096, 15.6515274, -20.6544685, 21.2061253
3: -5.9411888, 21.1416397, -5.5544548, 20.0211716, -25.9623604, 26.6960945
4: -4.8949690, 19.5565891, -4.5881572, 18.5381012, -23.4330711, 24.1447430

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5030330, upper bound: 27.4971425
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5177529, upper bound: 27.5021641
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.0815225, 14.3603363, -4.3949318, 15.3425512, -19.4240704, 18.7552662
1: -5.9201736, 14.7163811, -6.2909975, 15.7170744, -21.6372490, 21.0073719
2: -5.0029421, 16.5449162, -5.3071909, 17.6468716, -22.6498127, 21.8521061
3: -5.9411888, 21.1416397, -6.3698349, 22.6206188, -28.5618057, 27.5114746
4: -4.8949690, 19.5565891, -5.2139759, 20.9635792, -25.8585472, 24.7705593

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5030330, upper bound: 27.4971425
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5177528, upper bound: 27.5021641
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.7206116, 16.2323303, -3.8096104, 13.5872917, -18.3079014, 20.0419369
1: -6.7455416, 16.6543579, -5.5293055, 13.9026499, -20.6481915, 22.1836624
2: -5.7037611, 18.6685505, -4.6612096, 15.6515274, -21.3552837, 23.3297596
3: -6.8226271, 23.8968391, -5.5544548, 20.0211716, -26.8437996, 29.4512939
4: -5.5644245, 22.1408634, -4.5881572, 18.5381012, -24.1025238, 26.7290192

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5036455, upper bound: 27.5010190
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5183980, upper bound: 27.5052087
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.7206116, 16.2323303, -4.3976188, 15.3468742, -20.0674801, 20.6299458
1: -6.7455416, 16.6543579, -6.2940626, 15.7216339, -22.4671745, 22.9484215
2: -5.7037611, 18.6685505, -5.3095002, 17.6520042, -23.3557606, 23.9780502
3: -6.8226271, 23.8968391, -6.3730483, 22.6263657, -29.4489918, 30.2698879
4: -5.5644245, 22.1408634, -5.2157865, 20.9694633, -26.5338860, 27.3566494

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5036455, upper bound: 27.5015480
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5183980, upper bound: 27.5052087
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.0815225, 14.3603363, -4.0815225, 14.3603363, -18.4418564, 18.4418564
1: -5.9201736, 14.7163811, -5.9201736, 14.7163811, -20.6365509, 20.6365528
2: -5.0029421, 16.5449162, -5.0029421, 16.5449162, -21.5478573, 21.5478573
3: -5.9411888, 21.1416397, -5.9411888, 21.1416397, -27.0828247, 27.0828266
4: -4.8949690, 19.5565891, -4.8949690, 19.5565891, -24.4515572, 24.4515572

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5240779, upper bound: 27.5232443
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5373913, upper bound: 27.5373981
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.0815225, 14.3603363, -4.7206116, 16.2323303, -20.3138466, 19.0809460
1: -5.9201736, 14.7163811, -6.7455416, 16.6543579, -22.5745316, 21.4619217
2: -5.0029421, 16.5449162, -5.7037611, 18.6685505, -23.6714935, 22.2486744
3: -5.9411888, 21.1416397, -6.8226271, 23.8968391, -29.8380241, 27.9642639
4: -4.8949690, 19.5565891, -5.5644245, 22.1408634, -27.0358315, 25.1210098

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5240781, upper bound: 27.5232446
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4826132, upper bound: 27.5373981
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.7206116, 16.2323303, -4.0815225, 14.3603363, -19.0809479, 20.3138466
1: -6.7455416, 16.6543579, -5.9201736, 14.7163811, -21.4619198, 22.5745316
2: -5.7037611, 18.6685505, -5.0029421, 16.5449162, -22.2486744, 23.6714935
3: -6.8226271, 23.8968391, -5.9411888, 21.1416397, -27.9642639, 29.8380260
4: -5.5644245, 22.1408634, -4.8949690, 19.5565891, -25.1210098, 27.0358315

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4932297, upper bound: 27.4955512
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4826132, upper bound: 27.4851722
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.7206116, 16.2323303, -4.7206116, 16.2323303, -20.9529343, 20.9529324
1: -6.7455416, 16.6543579, -6.7455416, 16.6543579, -23.3998985, 23.3998985
2: -5.7037611, 18.6685505, -5.7037611, 18.6685505, -24.3723106, 24.3723106
3: -6.8226271, 23.8968391, -6.8226271, 23.8968391, -30.7194633, 30.7194653
4: -5.5644245, 22.1408634, -5.5644245, 22.1408634, -27.7052860, 27.7052860

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5261618, upper bound: 27.5370392
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5380429, upper bound: 27.5414581
time: 0.73 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.31 seconds
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5120794, upper bound: 27.5263943
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5021641, upper bound: 27.5177531
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5120794, upper bound: 27.5269993
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5021641, upper bound: 27.5183984
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5030330, upper bound: 27.4971425
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5177529, upper bound: 27.5021641
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5030330, upper bound: 27.4971425
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5177528, upper bound: 27.5021641
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5036455, upper bound: 27.5010190
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5183980, upper bound: 27.5052087
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5036455, upper bound: 27.5015480
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5183980, upper bound: 27.5052087
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5240779, upper bound: 27.5232443
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5373913, upper bound: 27.5373981
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5240781, upper bound: 27.5232446
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.4826132, upper bound: 27.5373981
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.4932297, upper bound: 27.4955512
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.4826132, upper bound: 27.4851722
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5261618, upper bound: 27.5370392
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5380429, upper bound: 27.5414581

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.5618644, 12.9171572, -4.0815225, 14.3603363, -17.9221992, 16.9986801
1: -5.1606417, 13.1869688, -5.9201736, 14.7163811, -19.8770161, 19.1071415
2: -4.3412910, 14.8576298, -5.0029421, 16.5449162, -20.8862076, 19.8605728
3: -5.1961222, 19.0373878, -5.9411888, 21.1416397, -26.3377590, 24.9785748
4: -4.2972727, 17.5959225, -4.8949690, 19.5565891, -23.8538628, 22.4908905

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4971425, upper bound: 27.5030330
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4826132, upper bound: 27.5177531
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.5618644, 12.9171572, -4.7206116, 16.2323303, -19.7941895, 17.6377659
1: -5.1606417, 13.1869688, -6.7455416, 16.6543579, -21.8149986, 19.9325104
2: -4.3412910, 14.8576298, -5.7037611, 18.6685505, -23.0098419, 20.5613899
3: -5.1961222, 19.0373878, -6.8226271, 23.8968391, -29.0929604, 25.8600121
4: -4.2972727, 17.5959225, -5.5644245, 22.1408634, -26.4381371, 23.1603451

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5010190, upper bound: 27.5036456
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5010190, upper bound: 27.5036456
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.8377326, 13.7042408, -4.0815225, 14.3603363, -18.1980686, 17.7857609
1: -5.5586572, 14.0156584, -5.9201736, 14.7163811, -20.2750340, 19.9358330
2: -4.6897707, 15.7574348, -5.0029421, 16.5449162, -21.2346878, 20.7603760
3: -5.5885458, 20.1776047, -5.9411888, 21.1416397, -26.7301846, 26.1187916
4: -4.6097212, 18.6277122, -4.8949690, 19.5565891, -24.1663094, 23.5226822

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5210527, upper bound: 27.5207850
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5210527, upper bound: 27.5207850
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.1616135, 14.7958717, -4.0664392, 14.3212748, -18.4828835, 18.8623104
1: -6.0217419, 15.1112652, -5.8990955, 14.6747179, -20.6964569, 21.0103550
2: -5.0678825, 16.9106216, -4.9846530, 16.4977531, -21.5656357, 21.8952751
3: -6.0606642, 21.6794033, -5.9208250, 21.0841732, -27.1448364, 27.6002254
4: -4.9605665, 20.0012703, -4.8784657, 19.5014000, -24.4619675, 24.8797359

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4826132, upper bound: 27.5260500
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5323766, upper bound: 27.5373979
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.8377326, 13.7042408, -4.7206116, 16.2323303, -20.0700569, 18.4248486
1: -5.5586572, 14.0156584, -6.7455416, 16.6543579, -22.2130146, 20.7612000
2: -4.6897707, 15.7574348, -5.7037611, 18.6685505, -23.3583221, 21.4611931
3: -5.5885458, 20.1776047, -6.8226271, 23.8968391, -29.4853859, 27.0002308
4: -4.6097212, 18.6277122, -5.5644245, 22.1408634, -26.7505836, 24.1921368

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5262105, upper bound: 27.5212535
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5262105, upper bound: 27.5239056
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.1616135, 14.7958717, -4.7037153, 16.1888542, -20.3504677, 19.4995861
1: -6.0217419, 15.1112652, -6.7219248, 16.6080399, -22.6297779, 21.8331871
2: -5.0678825, 16.9106216, -5.6835074, 18.6160450, -23.6839275, 22.5941257
3: -6.0606642, 21.6794033, -6.7995176, 23.8327618, -29.8934250, 28.4789181
4: -4.9605665, 20.0012703, -5.5462599, 22.0795708, -27.0401363, 25.5475311

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5384948, upper bound: 27.5273029
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5384948, upper bound: 27.5380641
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.4364772, 15.4927216, -4.7206116, 16.2323303, -20.6688080, 20.2133293
1: -6.3414073, 15.8659678, -6.7455416, 16.6543579, -22.9957657, 22.6115093
2: -5.3552303, 17.7862968, -5.7037611, 18.6685505, -24.0237808, 23.4900551
3: -6.4260044, 22.8168812, -6.8226271, 23.8968391, -30.3228436, 29.6395073
4: -5.2513251, 21.1006165, -5.5644245, 22.1408634, -27.3921871, 26.6650391

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5283592, upper bound: 27.5292450
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5283592, upper bound: 27.5370389
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.7756276, 16.5842285, -4.7037153, 16.1888542, -20.9644814, 21.2879429
1: -6.8313746, 16.9627132, -6.7219248, 16.6080399, -23.4394131, 23.6846371
2: -5.7417583, 18.9490433, -5.6835074, 18.6160450, -24.3578014, 24.6325474
3: -6.9129910, 24.3177433, -6.7995176, 23.8327618, -30.7457485, 31.1172581
4: -5.6055784, 22.4766064, -5.5462599, 22.0795708, -27.6851444, 28.0228653

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5375349, upper bound: 27.5308535
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5375349, upper bound: 27.5414579
time: 0.83 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.49 seconds
IS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.49
Output dim: 3, lower bound: -27.4971425, upper bound: 27.5030330
IS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.49
Output dim: 3, lower bound: -27.4826132, upper bound: 27.5177531
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.49
Output dim: 3, lower bound: -27.5010190, upper bound: 27.5036456
IS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.49
Output dim: 3, lower bound: -27.5010190, upper bound: 27.5036456
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -27.5210527, upper bound: 27.5207850
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -27.5210527, upper bound: 27.5207850
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -27.4826132, upper bound: 27.5260500
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -27.5323766, upper bound: 27.5373979
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -27.5262105, upper bound: 27.5212535
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -27.5262105, upper bound: 27.5239056
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -27.5384948, upper bound: 27.5273029
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -27.5384948, upper bound: 27.5380641
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -27.5283592, upper bound: 27.5292450
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -27.5283592, upper bound: 27.5370389
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -27.5375349, upper bound: 27.5308535
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -27.5375349, upper bound: 27.5414579

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.8377326, 13.7042408, -3.8377326, 13.7042408, -17.5419731, 17.5419731
1: -5.5586572, 14.0156584, -5.5586572, 14.0156584, -19.5743160, 19.5743160
2: -4.6897707, 15.7574348, -4.6897707, 15.7574348, -20.4472046, 20.4472046
3: -5.5885458, 20.1776047, -5.5885458, 20.1776047, -25.7661514, 25.7661514
4: -4.6097212, 18.6277122, -4.6097212, 18.6277122, -23.2374344, 23.2374344

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5089565, upper bound: 27.4967693
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4885081, upper bound: 27.4892980
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.8377326, 13.7042408, -4.1616135, 14.7958717, -18.6336040, 17.8658524
1: -5.5586572, 14.0156584, -6.0217419, 15.1112652, -20.6699181, 20.0373993
2: -4.6897707, 15.7574348, -5.0678825, 16.9106216, -21.6003914, 20.8253174
3: -5.5885458, 20.1776047, -6.0606642, 21.6794033, -27.2679482, 26.2382679
4: -4.6097212, 18.6277122, -4.9605665, 20.0012703, -24.6109924, 23.5882797

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5089564, upper bound: 27.5165136
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4885081, upper bound: 27.4948427
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.1616135, 14.7958717, -3.8377326, 13.7042408, -17.8658524, 18.6336040
1: -6.0217419, 15.1112652, -5.5586572, 14.0156584, -20.0373993, 20.6699181
2: -5.0678825, 16.9106216, -4.6897707, 15.7574348, -20.8253174, 21.6003914
3: -6.0606642, 21.6794033, -5.5885458, 20.1776047, -26.2382660, 27.2679482
4: -4.9605665, 20.0012703, -4.6097212, 18.6277122, -23.5882797, 24.6109924

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5206794, upper bound: 27.5033608
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5193409, upper bound: 27.5030354
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.1616135, 14.7958717, -4.1616135, 14.7958717, -18.9574833, 18.9574833
1: -6.0217419, 15.1112652, -6.0217419, 15.1112652, -21.1330032, 21.1330032
2: -5.0678825, 16.9106216, -5.0678825, 16.9106216, -21.9785042, 21.9785042
3: -6.0606642, 21.6794033, -6.0606642, 21.6794033, -27.7400665, 27.7400665
4: -4.9605665, 20.0012703, -4.9605665, 20.0012703, -24.9618378, 24.9618378

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5206798, upper bound: 27.5033608
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5193413, upper bound: 27.5030355
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.8377326, 13.7042408, -4.4364772, 15.4927216, -19.3304539, 18.1407185
1: -5.5586572, 14.0156584, -6.3414073, 15.8659678, -21.4246254, 20.3570652
2: -4.6897707, 15.7574348, -5.3552303, 17.7862968, -22.4760666, 21.1126652
3: -5.5885458, 20.1776047, -6.4260044, 22.8168812, -28.4054260, 26.6036091
4: -4.6097212, 18.6277122, -5.2513251, 21.1006165, -25.7103386, 23.8790379

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5234132, upper bound: 27.5163986
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4945134, upper bound: 27.5007293
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.8377326, 13.7042408, -4.7756276, 16.5842285, -20.4219608, 18.4798679
1: -5.5586572, 14.0156584, -6.8313746, 16.9627132, -22.5213699, 20.8470325
2: -4.6897707, 15.7574348, -5.7417583, 18.9490433, -23.6388130, 21.4991913
3: -5.5885458, 20.1776047, -6.9129910, 24.3177433, -29.9062881, 27.0905895
4: -4.6097212, 18.6277122, -5.6055784, 22.4766064, -27.0863266, 24.2332897

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5234132, upper bound: 27.5205268
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4945134, upper bound: 27.5007293
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.1616135, 14.7958717, -4.4364772, 15.4927216, -19.6543350, 19.2323494
1: -6.0217419, 15.1112652, -6.3414073, 15.8659678, -21.8877106, 21.4526691
2: -5.0678825, 16.9106216, -5.3552303, 17.7862968, -22.8541794, 22.2658520
3: -6.0606642, 21.6794033, -6.4260044, 22.8168812, -28.8775444, 28.1054077
4: -4.9605665, 20.0012703, -5.2513251, 21.1006165, -26.0611839, 25.2525940

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5362451, upper bound: 27.5242262
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5330160, upper bound: 27.5223326
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.1616135, 14.7958717, -4.7756276, 16.5842285, -20.7458382, 19.5714989
1: -6.0217419, 15.1112652, -6.8313746, 16.9627132, -22.9844551, 21.9426384
2: -5.0678825, 16.9106216, -5.7417583, 18.9490433, -24.0169258, 22.6523781
3: -6.0606642, 21.6794033, -6.9129910, 24.3177433, -30.3784065, 28.5923901
4: -4.9605665, 20.0012703, -5.6055784, 22.4766064, -27.4371719, 25.6068459

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5362454, upper bound: 27.5357266
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5330162, upper bound: 27.5350867
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.4364772, 15.4927216, -4.4364772, 15.4927216, -19.9291992, 19.9291992
1: -6.3414073, 15.8659678, -6.3414073, 15.8659678, -22.2073746, 22.2073746
2: -5.3552303, 17.7862968, -5.3552303, 17.7862968, -23.1415253, 23.1415253
3: -6.4260044, 22.8168812, -6.4260044, 22.8168812, -29.2428856, 29.2428856
4: -5.2513251, 21.1006165, -5.2513251, 21.1006165, -26.3519402, 26.3519402

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5239665, upper bound: 27.5232775
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5264857, upper bound: 27.5266107
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.4364772, 15.4927216, -4.7756276, 16.5842285, -21.0207062, 20.2683487
1: -6.3414073, 15.8659678, -6.8313746, 16.9627132, -23.3041191, 22.6973419
2: -5.3552303, 17.7862968, -5.7417583, 18.9490433, -24.3042717, 23.5280514
3: -6.4260044, 22.8168812, -6.9129910, 24.3177433, -30.7437477, 29.7298679
4: -5.2513251, 21.1006165, -5.6055784, 22.4766064, -27.7279301, 26.7061920

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5239665, upper bound: 27.5232775
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4934568, upper bound: 27.5348855
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.7756276, 16.5842285, -4.4364772, 15.4927216, -20.2683487, 21.0207062
1: -6.8313746, 16.9627132, -6.3414073, 15.8659678, -22.6973419, 23.3041191
2: -5.7417583, 18.9490433, -5.3552303, 17.7862968, -23.5280533, 24.3042717
3: -6.9129910, 24.3177433, -6.4260044, 22.8168812, -29.7298698, 30.7437477
4: -5.6055784, 22.4766064, -5.2513251, 21.1006165, -26.7061920, 27.7279301

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5306385, upper bound: 27.5251323
time: 1.66 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5355882, upper bound: 27.5284547
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.7756276, 16.5842285, -4.7756276, 16.5842285, -21.3598537, 21.3598537
1: -6.8313746, 16.9627132, -6.8313746, 16.9627132, -23.7940884, 23.7940884
2: -5.7417583, 18.9490433, -5.7417583, 18.9490433, -24.6907997, 24.6907978
3: -6.9129910, 24.3177433, -6.9129910, 24.3177433, -31.2307301, 31.2307301
4: -5.6055784, 22.4766064, -5.6055784, 22.4766064, -28.0821819, 28.0821819

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5306386, upper bound: 27.5251324
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5355885, upper bound: 27.5284548
time: 0.97 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.52 seconds
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.5089565, upper bound: 27.4967693
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.4885081, upper bound: 27.4892980
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.5089564, upper bound: 27.5165136
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.4885081, upper bound: 27.4948427
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.5206794, upper bound: 27.5033608
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.5193409, upper bound: 27.5030354
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.5206798, upper bound: 27.5033608
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.5193413, upper bound: 27.5030355
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.5234132, upper bound: 27.5163986
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.4945134, upper bound: 27.5007293
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.5234132, upper bound: 27.5205268
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.4945134, upper bound: 27.5007293
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.5362451, upper bound: 27.5242262
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.5330160, upper bound: 27.5223326
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.5362454, upper bound: 27.5357266
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.5330162, upper bound: 27.5350867
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.5239665, upper bound: 27.5232775
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.5264857, upper bound: 27.5266107
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.5239665, upper bound: 27.5232775
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.4934568, upper bound: 27.5348855
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.5306385, upper bound: 27.5251323
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.5355882, upper bound: 27.5284547
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.5306386, upper bound: 27.5251324
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 3, lower bound: -27.5355885, upper bound: 27.5284548

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -3.7654068, 13.5031567, -17.1496143, 17.1155968
1: -5.2464833, 13.5860958, -5.4479327, 13.8041906, -19.0506725, 19.0340290
2: -4.3717928, 15.2154961, -4.5926566, 15.5196924, -19.8914814, 19.8081532
3: -5.2993393, 19.6015205, -5.4813008, 19.8881969, -25.1875362, 25.0828209
4: -4.3268204, 18.0051994, -4.5222597, 18.3482151, -22.6750355, 22.5274582

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5193411, upper bound: 27.5030354
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5193411, upper bound: 27.5030354
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -4.0913639, 14.6009912, -18.2474499, 17.4415550
1: -5.2464833, 13.5860958, -5.9159212, 14.9066668, -20.1531467, 19.5020180
2: -4.3717928, 15.2154961, -4.9736328, 16.6819992, -21.0537891, 20.1891289
3: -5.2993393, 19.6015205, -5.9570332, 21.3985386, -26.6978779, 25.5585518
4: -4.3268204, 18.0051994, -4.8756719, 19.7307777, -24.0575981, 22.8808689

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5330831, upper bound: 27.5318199
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5330831, upper bound: 27.5318200
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.3112977, 12.2320118, -4.3616943, 15.2910557, -18.6023521, 16.5937023
1: -4.7633290, 12.4608803, -6.2309623, 15.6551294, -20.4184551, 18.6918430
2: -3.9792581, 14.0205498, -5.2576017, 17.5502262, -21.5294819, 19.2781525
3: -4.8087497, 18.0676193, -6.3205066, 22.5223827, -27.3311329, 24.3881264
4: -3.9650612, 16.5860634, -5.1631875, 20.8199844, -24.7850456, 21.7492504

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4971796, upper bound: 27.5017959
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4971796, upper bound: 27.5032107
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.3112977, 12.2320118, -4.7030725, 16.3897057, -19.7010040, 16.9350853
1: -4.7633290, 12.4608803, -6.7281890, 16.7596378, -21.5229645, 19.1890678
2: -3.9792581, 14.0205498, -5.6475244, 18.7224140, -22.7016697, 19.6680737
3: -4.8087497, 18.0676193, -6.8132281, 24.0338974, -28.8426476, 24.8808479
4: -3.9650612, 16.5860634, -5.5206356, 22.2078400, -26.1729012, 22.1066990

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5006678, upper bound: 27.5051647
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5235975, upper bound: 27.5191663
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -4.3616943, 15.2910557, -18.9375114, 17.7118855
1: -5.2464833, 13.5860958, -6.2309623, 15.6551294, -20.9016056, 19.8170586
2: -4.3717928, 15.2154961, -5.2576017, 17.5502262, -21.9220181, 20.4730988
3: -5.2993393, 19.6015205, -6.3205066, 22.5223827, -27.8217220, 25.9220257
4: -4.3268204, 18.0051994, -5.1631875, 20.8199844, -25.1468048, 23.1683826

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5315072, upper bound: 27.5222730
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5315070, upper bound: 27.5222729
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.1191621, 14.6685381, -4.4192700, 15.4461288, -19.5652905, 19.0878086
1: -5.9215693, 14.9546156, -6.3183599, 15.8169651, -21.7385254, 21.2729759
2: -4.9671373, 16.7323151, -5.3352942, 17.7330418, -22.7001762, 22.0676098
3: -5.9684920, 21.4562817, -6.4028993, 22.7512589, -28.7197475, 27.8591766
4: -4.8678341, 19.7656593, -5.2337227, 21.0376129, -25.9054470, 24.9993820

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5315072, upper bound: 27.5222730
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5315072, upper bound: 27.5223326
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -4.7030725, 16.3897057, -20.0361633, 18.0532627
1: -5.2464833, 13.5860958, -6.7281890, 16.7596378, -22.0061150, 20.3142853
2: -4.3717928, 15.2154961, -5.6475244, 18.7224140, -23.0942039, 20.8630199
3: -5.2993393, 19.6015205, -6.8132281, 24.0338974, -29.3332367, 26.4147472
4: -4.3268204, 18.0051994, -5.5206356, 22.2078400, -26.5346603, 23.5258350

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5346909, upper bound: 27.5331814
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5346910, upper bound: 27.5350865
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.1191621, 14.6685381, -4.7557540, 16.5309963, -20.6501579, 19.4242916
1: -5.9215693, 14.9546156, -6.8032150, 16.9067249, -22.8282890, 21.7578316
2: -4.9671373, 16.7323151, -5.7187605, 18.8876667, -23.8548050, 22.4510765
3: -5.9684920, 21.4562817, -6.8857288, 24.2424297, -30.2109184, 28.3420086
4: -4.8678341, 19.7656593, -5.5851955, 22.4043961, -27.2722301, 25.3508549

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5346909, upper bound: 27.5331814
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5346908, upper bound: 27.5350863
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.8882565, 14.0178547, -4.3616943, 15.2910557, -19.1793041, 18.3795490
1: -5.5631576, 14.3240805, -6.2309623, 15.6551294, -21.2182808, 20.5550423
2: -4.6534100, 16.0694389, -5.2576017, 17.5502262, -22.2036362, 21.3270397
3: -5.6674542, 20.6948471, -6.3205066, 22.5223827, -28.1898365, 27.0153542
4: -4.6273518, 19.0684605, -5.1631875, 20.8199844, -25.4473362, 24.2316456

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5231677, upper bound: 27.5232775
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5231677, upper bound: 27.5232775
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.4765372, 15.6225147, -4.4192700, 15.4461288, -19.9226665, 20.0417843
1: -6.3536415, 15.9724932, -6.3183599, 15.8169651, -22.1706009, 22.2908535
2: -5.3602648, 17.9039478, -5.3352942, 17.7330418, -23.0933056, 23.2392426
3: -6.4377475, 22.9774036, -6.4028993, 22.7512589, -29.1890068, 29.3802986
4: -5.2538066, 21.2279301, -5.2337227, 21.0376129, -26.2914181, 26.4616528

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5236471, upper bound: 27.5263080
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5040078, upper bound: 27.5043695
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.8882565, 14.0178547, -4.7030725, 16.3897057, -20.2779579, 18.7209282
1: -5.5631576, 14.3240805, -6.7281890, 16.7596378, -22.3227901, 21.0522690
2: -4.6534100, 16.0694389, -5.6475244, 18.7224140, -23.3758240, 21.7169628
3: -5.6674542, 20.6948471, -6.8132281, 24.0338974, -29.7013512, 27.5080757
4: -4.6273518, 19.0684605, -5.5206356, 22.2078400, -26.8351917, 24.5890961

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5240071, upper bound: 27.5304998
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5240071, upper bound: 27.5318968
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.4765372, 15.6225147, -4.7557540, 16.5309963, -21.0075321, 20.3782673
1: -6.3536415, 15.9724932, -6.8032150, 16.9067249, -23.2603626, 22.7757072
2: -5.3602648, 17.9039478, -5.7187605, 18.8876667, -24.2479324, 23.6227074
3: -6.4377475, 22.9774036, -6.8857288, 24.2424297, -30.6801777, 29.8631325
4: -5.2538066, 21.2279301, -5.5851955, 22.4043961, -27.6581993, 26.8131256

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5252265, upper bound: 27.5337064
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5252265, upper bound: 27.5348858
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.2396750, 15.1407347, -4.3616943, 15.2910557, -19.5307255, 19.5024281
1: -6.0757089, 15.4544592, -6.2309623, 15.6551294, -21.7308311, 21.6854210
2: -5.0678725, 17.2726898, -5.2576017, 17.5502262, -22.6180973, 22.5302925
3: -6.1715178, 22.2449474, -6.3205066, 22.5223827, -28.6939011, 28.5654545
4: -5.0068088, 20.4940319, -5.1631875, 20.8199844, -25.8267937, 25.6572170

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5306386, upper bound: 27.5251323
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5283093, upper bound: 27.5117140
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5303626, upper bound: 27.5248340
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.8151884, 16.7127552, -4.4192700, 15.4461288, -20.2613163, 21.1320248
1: -6.8487349, 17.0677147, -6.3183599, 15.8169651, -22.6656952, 23.3860741
2: -5.7475977, 19.0672207, -5.3352942, 17.7330418, -23.4806366, 24.4025154
3: -6.9301233, 24.4769001, -6.4028993, 22.7512589, -29.6813812, 30.8797951
4: -5.6081338, 22.6030693, -5.2337227, 21.0376129, -26.6457462, 27.8367920

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4912101, upper bound: 27.4900863
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5347068, upper bound: 27.5284547
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.2396750, 15.1407347, -4.7030725, 16.3897057, -20.6293755, 19.8438072
1: -6.0757089, 15.4544592, -6.7281890, 16.7596378, -22.8353405, 22.1826477
2: -5.0678725, 17.2726898, -5.6475244, 18.7224140, -23.7902851, 22.9202137
3: -6.1715178, 22.2449474, -6.8132281, 24.0338974, -30.2054157, 29.0581760
4: -5.0068088, 20.4940319, -5.5206356, 22.2078400, -27.2146492, 26.0146675

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5358327, upper bound: 27.5352757
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5358327, upper bound: 27.5355267
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.8151884, 16.7127552, -4.7557540, 16.5309963, -21.3461819, 21.4685097
1: -6.8487349, 17.0677147, -6.8032150, 16.9067249, -23.7554569, 23.8709278
2: -5.7475977, 19.0672207, -5.7187605, 18.8876667, -24.6352634, 24.7859802
3: -6.9301233, 24.4769001, -6.8857288, 24.2424297, -31.1725540, 31.3626289
4: -5.6081338, 22.6030693, -5.5851955, 22.4043961, -28.0125294, 28.1882648

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5371360, upper bound: 27.5362609
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5289300, upper bound: 27.5227600
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5160982, upper bound: 27.4998171
time: 0.88 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 6.13 seconds
IS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5193411, upper bound: 27.5030354
IS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5193411, upper bound: 27.5030354
IS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5330831, upper bound: 27.5318199
IS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5330831, upper bound: 27.5318200
IS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.4971796, upper bound: 27.5017959
IS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.4971796, upper bound: 27.5032107
IS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5006678, upper bound: 27.5051647
IS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5235975, upper bound: 27.5191663
IS_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5315072, upper bound: 27.5222730
IS_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5315070, upper bound: 27.5222729
IS_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5315072, upper bound: 27.5222730
IS_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5315072, upper bound: 27.5223326
IS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5346909, upper bound: 27.5331814
IS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5346910, upper bound: 27.5350865
IS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5346909, upper bound: 27.5331814
IS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5346908, upper bound: 27.5350863
IS_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5231677, upper bound: 27.5232775
IS_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5231677, upper bound: 27.5232775
IS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5236471, upper bound: 27.5263080
IS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5040078, upper bound: 27.5043695
IS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5240071, upper bound: 27.5304998
IS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5240071, upper bound: 27.5318968
IS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5252265, upper bound: 27.5337064
IS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5252265, upper bound: 27.5348858
IS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5283093, upper bound: 27.5117140
IS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5303626, upper bound: 27.5248340
IS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.4912101, upper bound: 27.4900863
IS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5347068, upper bound: 27.5284547
IS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5358327, upper bound: 27.5352757
IS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5358327, upper bound: 27.5355267
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5289300, upper bound: 27.5227600
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 6.13
Output dim: 3, lower bound: -27.5160982, upper bound: 27.4998171

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -3.6464579, 13.3501921, -16.9966488, 16.9966488
1: -5.2464833, 13.5860958, -5.2464833, 13.5860958, -18.8325787, 18.8325787
2: -4.3717928, 15.2154961, -4.3717928, 15.2154961, -19.5872879, 19.5872879
3: -5.2993393, 19.6015205, -5.2993393, 19.6015205, -24.9008598, 24.9008598
4: -4.3268204, 18.0051994, -4.3268204, 18.0051994, -22.3320179, 22.3320179

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5224149, upper bound: 27.5304944
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5341611, upper bound: 27.5316254
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -4.1191621, 14.6685381, -18.3149967, 17.4693546
1: -5.2464833, 13.5860958, -5.9215693, 14.9546156, -20.2010994, 19.5076656
2: -4.3717928, 15.2154961, -4.9671373, 16.7323151, -21.1041069, 20.1826324
3: -5.2993393, 19.6015205, -5.9684920, 21.4562817, -26.7556190, 25.5700092
4: -4.3268204, 18.0051994, -4.8678341, 19.7656593, -24.0924797, 22.8730335

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5224149, upper bound: 27.5304943
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5341611, upper bound: 27.5316254
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.3112977, 12.2320118, -4.6852412, 16.3467007, -19.6579990, 16.9172478
1: -4.7633290, 12.4608803, -6.7039127, 16.7155361, -21.4788647, 19.1647930
2: -3.9792581, 14.0205498, -5.6284022, 18.6725655, -22.6518230, 19.6489525
3: -4.8087497, 18.0676193, -6.7906041, 23.9733906, -28.7821407, 24.8582230
4: -3.9650612, 16.5860634, -5.5038037, 22.1498432, -26.1149044, 22.0898666

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4887731, upper bound: 27.5022851
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5235670, upper bound: 27.5186312
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -3.8882565, 14.0178547, -17.6643124, 17.2384453
1: -5.2464833, 13.5860958, -5.5631576, 14.3240805, -19.5705624, 19.1492538
2: -4.3717928, 15.2154961, -4.6534100, 16.0694389, -20.4412289, 19.8689060
3: -5.2993393, 19.6015205, -5.6674542, 20.6948471, -25.9941864, 25.2689743
4: -4.3268204, 18.0051994, -4.6273518, 19.0684605, -23.3952808, 22.6325512

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4998402, upper bound: 27.4938748
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4924928, upper bound: 27.4758690
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -4.4765372, 15.6225147, -19.2689705, 17.8267288
1: -5.2464833, 13.5860958, -6.3536415, 15.9724932, -21.2189770, 19.9397373
2: -4.3717928, 15.2154961, -5.3602648, 17.9039478, -22.2757397, 20.5757599
3: -5.2993393, 19.6015205, -6.4377475, 22.9774036, -28.2767410, 26.0392685
4: -4.3268204, 18.0051994, -5.2538066, 21.2279301, -25.5547504, 23.2590027

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4998402, upper bound: 27.4966005
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4924929, upper bound: 27.4763273
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.1191621, 14.6685381, -3.8882565, 14.0178547, -18.1370163, 18.5567932
1: -5.9215693, 14.9546156, -5.5631576, 14.3240805, -20.2456493, 20.5177727
2: -4.9671373, 16.7323151, -4.6534100, 16.0694389, -21.0365753, 21.3857250
3: -5.9684920, 21.4562817, -5.6674542, 20.6948471, -26.6633358, 27.1237354
4: -4.8678341, 19.7656593, -4.6273518, 19.0684605, -23.9362946, 24.3930111

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5216421, upper bound: 27.5214542
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5311017, upper bound: 27.5221103
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.1191621, 14.6685381, -4.4765372, 15.6225147, -19.7416763, 19.1450748
1: -5.9215693, 14.9546156, -6.3536415, 15.9724932, -21.8940620, 21.3082581
2: -4.9671373, 16.7323151, -5.3602648, 17.9039478, -22.8710861, 22.0925789
3: -5.9684920, 21.4562817, -6.4377475, 22.9774036, -28.9458904, 27.8940277
4: -4.8678341, 19.7656593, -5.2538066, 21.2279301, -26.0957642, 25.0194645

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5216420, upper bound: 27.5214546
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5311015, upper bound: 27.5221102
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -4.2396750, 15.1407347, -18.7871933, 17.5898647
1: -5.2464833, 13.5860958, -6.0757089, 15.4544592, -20.7009392, 19.6618042
2: -4.3717928, 15.2154961, -5.0678725, 17.2726898, -21.6444817, 20.2833691
3: -5.2993393, 19.6015205, -6.1715178, 22.2449474, -27.5442867, 25.7730389
4: -4.3268204, 18.0051994, -5.0068088, 20.4940319, -24.8208523, 23.0120087

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5087325, upper bound: 27.5045085
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4990864, upper bound: 27.4827818
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -4.8151884, 16.7127552, -20.3592129, 18.1653767
1: -5.2464833, 13.5860958, -6.8487349, 17.0677147, -22.3141975, 20.4348297
2: -4.3717928, 15.2154961, -5.7475977, 19.0672207, -23.4390125, 20.9630890
3: -5.2993393, 19.6015205, -6.9301233, 24.4769001, -29.7762394, 26.5316429
4: -4.3268204, 18.0051994, -5.6081338, 22.6030693, -26.9298878, 23.6133327

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5087324, upper bound: 27.5086857
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4990864, upper bound: 27.4828645
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.1191621, 14.6685381, -4.2396750, 15.1407347, -19.2598972, 18.9082127
1: -5.9215693, 14.9546156, -6.0757089, 15.4544592, -21.3760262, 21.0303249
2: -4.9671373, 16.7323151, -5.0678725, 17.2726898, -22.2398262, 21.8001881
3: -5.9684920, 21.4562817, -6.1715178, 22.2449474, -28.2134361, 27.6278000
4: -4.8678341, 19.7656593, -5.0068088, 20.4940319, -25.3618660, 24.7724686

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5248300, upper bound: 27.5322884
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5342854, upper bound: 27.5329038
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.1191621, 14.6685381, -4.8151884, 16.7127552, -20.8319168, 19.4837246
1: -5.9215693, 14.9546156, -6.8487349, 17.0677147, -22.9892807, 21.8033504
2: -4.9671373, 16.7323151, -5.7475977, 19.0672207, -24.0343571, 22.4799099
3: -5.9684920, 21.4562817, -6.9301233, 24.4769001, -30.4453888, 28.3864059
4: -4.8678341, 19.7656593, -5.6081338, 22.6030693, -27.4709034, 25.3737926

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5248300, upper bound: 27.5322884
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5342856, upper bound: 27.5329041
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.8882565, 14.0178547, -3.8882565, 14.0178547, -17.9061089, 17.9061089
1: -5.5631576, 14.3240805, -5.5631576, 14.3240805, -19.8872356, 19.8872356
2: -4.6534100, 16.0694389, -4.6534100, 16.0694389, -20.7228489, 20.7228489
3: -5.6674542, 20.6948471, -5.6674542, 20.6948471, -26.3623009, 26.3623009
4: -4.6273518, 19.0684605, -4.6273518, 19.0684605, -23.6958122, 23.6958122

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5163825, upper bound: 27.5230258
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5206131, upper bound: 27.5206539
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.8882565, 14.0178547, -4.4765372, 15.6225147, -19.5107651, 18.4943924
1: -5.5631576, 14.3240805, -6.3536415, 15.9724932, -21.5356503, 20.6777210
2: -4.6534100, 16.0694389, -5.3602648, 17.9039478, -22.5573578, 21.4297009
3: -5.6674542, 20.6948471, -6.4377475, 22.9774036, -28.6448574, 27.1325951
4: -4.6273518, 19.0684605, -5.2538066, 21.2279301, -25.8552818, 24.3222656

Time for backsubstitution: 2.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5163825, upper bound: 27.5230364
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5163825, upper bound: 27.5207305
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.4765372, 15.6225147, -3.8882565, 14.0178547, -18.4943924, 19.5107651
1: -6.3536415, 15.9724932, -5.5631576, 14.3240805, -20.6777210, 21.5356503
2: -5.3602648, 17.9039478, -4.6534100, 16.0694389, -21.4297009, 22.5573578
3: -6.4377475, 22.9774036, -5.6674542, 20.6948471, -27.1325951, 28.6448574
4: -5.2538066, 21.2279301, -4.6273518, 19.0684605, -24.3222656, 25.8552818

Time for backsubstitution: 2.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5175703, upper bound: 27.5251850
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5233213, upper bound: 27.5261421
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.8882565, 14.0178547, -4.2396750, 15.1407347, -19.0289860, 18.2575302
1: -5.5631576, 14.3240805, -6.0757089, 15.4544592, -21.0176125, 20.3997860
2: -4.6534100, 16.0694389, -5.0678725, 17.2726898, -21.9260998, 21.1373100
3: -5.6674542, 20.6948471, -6.1715178, 22.2449474, -27.9124012, 26.8663654
4: -4.6273518, 19.0684605, -5.0068088, 20.4940319, -25.1213837, 24.0752697

Time for backsubstitution: 2.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5179896, upper bound: 27.5303457
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5222745, upper bound: 27.5278047
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.8882565, 14.0178547, -4.8151884, 16.7127552, -20.6010113, 18.8330402
1: -5.5631576, 14.3240805, -6.8487349, 17.0677147, -22.6308708, 21.1728153
2: -4.6534100, 16.0694389, -5.7475977, 19.0672207, -23.7206306, 21.8170319
3: -5.6674542, 20.6948471, -6.9301233, 24.4769001, -30.1443539, 27.6249695
4: -4.6273518, 19.0684605, -5.6081338, 22.6030693, -27.2304211, 24.6765938

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5179897, upper bound: 27.5317299
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5222744, upper bound: 27.5278045
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.4765372, 15.6225147, -4.2396750, 15.1407347, -19.6172714, 19.8621864
1: -6.3536415, 15.9724932, -6.0757089, 15.4544592, -21.8080997, 22.0482006
2: -5.3602648, 17.9039478, -5.0678725, 17.2726898, -22.6329536, 22.9718208
3: -6.4377475, 22.9774036, -6.1715178, 22.2449474, -28.6826954, 29.1489220
4: -5.2538066, 21.2279301, -5.0068088, 20.4940319, -25.7478371, 26.2347393

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5184059, upper bound: 27.5314126
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5249386, upper bound: 27.5333894
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.4765372, 15.6225147, -4.8151884, 16.7127552, -21.1892929, 20.4376984
1: -6.3536415, 15.9724932, -6.8487349, 17.0677147, -23.4213562, 22.8212280
2: -5.3602648, 17.9039478, -5.7475977, 19.0672207, -24.4274845, 23.6515427
3: -6.4377475, 22.9774036, -6.9301233, 24.4769001, -30.9146481, 29.9075279
4: -5.2538066, 21.2279301, -5.6081338, 22.6030693, -27.8568726, 26.8360634

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5184058, upper bound: 27.5325318
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5249386, upper bound: 27.5345394
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.2099819, 15.0545835, -4.4116035, 15.4257202, -19.6357021, 19.4661865
1: -6.0341425, 15.3639421, -6.2914410, 15.7849360, -21.8190765, 21.6553841
2: -5.0320191, 17.1727810, -5.3008504, 17.6865959, -22.7186127, 22.4736290
3: -6.1294856, 22.1190701, -6.3810730, 22.6976318, -28.8271160, 28.5001431
4: -4.9741917, 20.3766880, -5.1986890, 20.9978046, -25.9719963, 25.5753765

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5129379, upper bound: 27.4983462
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5168809, upper bound: 27.4945824
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.2396750, 15.1407347, -4.2767777, 15.0546541, -19.2943287, 19.4175129
1: -6.0757089, 15.4544592, -6.1104922, 15.4060669, -21.4817715, 21.5649471
2: -5.0678725, 17.2726898, -5.1531377, 17.2742786, -22.3421497, 22.4258270
3: -6.1715178, 22.2449474, -6.2003226, 22.1787224, -28.3502407, 28.4452705
4: -5.0068088, 20.4940319, -5.0702558, 20.4922619, -25.4990711, 25.5642872

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4930026, upper bound: 27.5010385
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4957908, upper bound: 27.4959759
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.8151884, 16.7127552, -4.4765372, 15.6225147, -20.4376984, 21.1892929
1: -6.8487349, 17.0677147, -6.3536415, 15.9724932, -22.8212280, 23.4213562
2: -5.7475977, 19.0672207, -5.3602648, 17.9039478, -23.6515427, 24.4274845
3: -6.9301233, 24.4769001, -6.4377475, 22.9774036, -29.9075279, 30.9146481
4: -5.6081338, 22.6030693, -5.2538066, 21.2279301, -26.8360634, 27.8568726

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5200644, upper bound: 27.5086635
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5341136, upper bound: 27.5278002
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.2396750, 15.1407347, -4.2396750, 15.1407347, -19.3804073, 19.3804073
1: -6.0757089, 15.4544592, -6.0757089, 15.4544592, -21.5301647, 21.5301647
2: -5.0678725, 17.2726898, -5.0678725, 17.2726898, -22.3405628, 22.3405628
3: -6.1715178, 22.2449474, -6.1715178, 22.2449474, -28.4164658, 28.4164658
4: -5.0068088, 20.4940319, -5.0068088, 20.4940319, -25.5008411, 25.5008411

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5062530, upper bound: 27.5093575
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5062530, upper bound: 27.5062530
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.2396750, 15.1407347, -4.8151884, 16.7127552, -20.9524307, 19.9559193
1: -6.0757089, 15.4544592, -6.8487349, 17.0677147, -23.1434212, 22.3031940
2: -5.0678725, 17.2726898, -5.7475977, 19.0672207, -24.1350918, 23.0202866
3: -6.1715178, 22.2449474, -6.9301233, 24.4769001, -30.6484184, 29.1750717
4: -5.0068088, 20.4940319, -5.6081338, 22.6030693, -27.6098785, 26.1021652

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5062530, upper bound: 27.5093589
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5062530, upper bound: 27.5062530
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.5564513, 15.9359255, -3.3289030, 12.2145786, -16.7710266, 19.2648277
1: -6.4719539, 16.2474136, -4.7385902, 12.3944864, -18.8664398, 20.9860039
2: -5.4259682, 18.1564064, -3.9912033, 13.8433456, -19.2693138, 22.1476078
3: -6.5360055, 23.3364792, -4.7493286, 17.8844566, -24.4204617, 28.0858078
4: -5.3091087, 21.5378685, -3.9541197, 16.4513321, -21.7604389, 25.4919853

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5223967, upper bound: 27.5204887
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4605628, upper bound: 27.4886090
time: 0.85 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 4.67 seconds
IS_A2_B2_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5224149, upper bound: 27.5304944
IS_A2_B2_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5341611, upper bound: 27.5316254
IS_A2_B2_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5224149, upper bound: 27.5304943
IS_A2_B2_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5341611, upper bound: 27.5316254
IS_A2_B2_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.4887731, upper bound: 27.5022851
IS_A2_B2_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5235670, upper bound: 27.5186312
IS_A2_B2_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.4998402, upper bound: 27.4938748
IS_A2_B2_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.4924928, upper bound: 27.4758690
IS_A2_B2_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.4998402, upper bound: 27.4966005
IS_A2_B2_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.4924929, upper bound: 27.4763273
IS_A2_B2_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5216421, upper bound: 27.5214542
IS_A2_B2_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5311017, upper bound: 27.5221103
IS_A2_B2_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5216420, upper bound: 27.5214546
IS_A2_B2_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5311015, upper bound: 27.5221102
IS_A2_B2_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5087325, upper bound: 27.5045085
IS_A2_B2_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.4990864, upper bound: 27.4827818
IS_A2_B2_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5087324, upper bound: 27.5086857
IS_A2_B2_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.4990864, upper bound: 27.4828645
IS_A2_B2_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5248300, upper bound: 27.5322884
IS_A2_B2_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5342854, upper bound: 27.5329038
IS_A2_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5248300, upper bound: 27.5322884
IS_A2_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5342856, upper bound: 27.5329041
IS_A2_B2_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5163825, upper bound: 27.5230258
IS_A2_B2_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5206131, upper bound: 27.5206539
IS_A2_B2_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5163825, upper bound: 27.5230364
IS_A2_B2_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5163825, upper bound: 27.5207305
IS_A2_B2_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5175703, upper bound: 27.5251850
IS_A2_B2_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5233213, upper bound: 27.5261421
IS_A2_B2_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5179896, upper bound: 27.5303457
IS_A2_B2_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5222745, upper bound: 27.5278047
IS_A2_B2_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5179897, upper bound: 27.5317299
IS_A2_B2_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5222744, upper bound: 27.5278045
IS_A2_B2_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5184059, upper bound: 27.5314126
IS_A2_B2_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5249386, upper bound: 27.5333894
IS_A2_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5184058, upper bound: 27.5325318
IS_A2_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5249386, upper bound: 27.5345394
IS_A2_B2_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5129379, upper bound: 27.4983462
IS_A2_B2_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5168809, upper bound: 27.4945824
IS_A2_B2_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.4930026, upper bound: 27.5010385
IS_A2_B2_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.4957908, upper bound: 27.4959759
IS_A2_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5200644, upper bound: 27.5086635
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5341136, upper bound: 27.5278002
IS_A2_B2_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5062530, upper bound: 27.5093575
IS_A2_B2_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5062530, upper bound: 27.5062530
IS_A2_B2_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5062530, upper bound: 27.5093589
IS_A2_B2_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5062530, upper bound: 27.5062530
IS_A2_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.5223967, upper bound: 27.5204887
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 3, lower bound: -27.4605628, upper bound: 27.4886090

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.6164749, 13.2819958, -3.6125371, 13.2505856, -16.8670559, 16.8945312
1: -5.1846161, 13.4975052, -5.1968441, 13.4815397, -18.6661568, 18.6943474
2: -4.3129835, 15.1071482, -4.3288002, 15.0992928, -19.4122753, 19.4359455
3: -5.2438045, 19.4920826, -5.2500544, 19.4558792, -24.6996841, 24.7421379
4: -4.2717013, 17.8941593, -4.2879238, 17.8679619, -22.1396618, 22.1820831

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5233782, upper bound: 27.5233782
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5233782, upper bound: 27.5322873
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -3.5840602, 13.1680984, -3.6464579, 13.3501921, -16.9342518, 16.8145561
1: -5.1535125, 13.3938541, -5.2464833, 13.5860958, -18.7396069, 18.6403370
2: -4.2921443, 15.0030842, -4.3717928, 15.2154961, -19.5076408, 19.3748779
3: -5.2045627, 19.3389301, -5.2993393, 19.6015205, -24.8060837, 24.6382694
4: -4.2549839, 17.7536049, -4.3268204, 18.0051994, -22.2601814, 22.0804234

Time for backsubstitution: 2.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5057843, upper bound: 27.4873370
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4754364, upper bound: 27.4754364
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.6164749, 13.2819958, -4.0858645, 14.5701828, -18.1866531, 17.3678570
1: -5.1846161, 13.4975052, -5.8726058, 14.8511114, -20.0357227, 19.3701115
2: -4.3129835, 15.1071482, -4.9246607, 16.6175308, -20.9305134, 20.0318089
3: -5.2438045, 19.4920826, -5.9194250, 21.3123894, -26.5561943, 25.4115067
4: -4.2717013, 17.8941593, -4.8293986, 19.6301117, -23.9018135, 22.7235584

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5128138, upper bound: 27.5202738
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5199470, upper bound: 27.5283393
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.5840602, 13.1680984, -4.1191621, 14.6685381, -18.2525978, 17.2872601
1: -5.1535125, 13.3938541, -5.9215693, 14.9546156, -20.1081276, 19.3154240
2: -4.2921443, 15.0030842, -4.9671373, 16.7323151, -21.0244598, 19.9702225
3: -5.2045627, 19.3389301, -5.9684920, 21.4562817, -26.6608429, 25.3074188
4: -4.2549839, 17.7536049, -4.8678341, 19.7656593, -24.0206432, 22.6214390

Time for backsubstitution: 2.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5336110, upper bound: 27.5228199
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5336110, upper bound: 27.5316258
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.2503290, 12.0559664, -4.6852412, 16.3467007, -19.5970306, 16.7412052
1: -4.6728878, 12.2730789, -6.7039127, 16.7155361, -21.3884239, 18.9769917
2: -3.9022455, 13.8133469, -5.6284022, 18.6725655, -22.5748100, 19.4417496
3: -4.7164497, 17.8124847, -6.7906041, 23.9733906, -28.6898403, 24.6030884
4: -3.8956451, 16.3417244, -5.5038037, 22.1498432, -26.0454884, 21.8455257

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5216605, upper bound: 27.5173344
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5216605, upper bound: 27.5173344
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.1237140, 14.6792870, -3.8583958, 13.9290609, -18.0527744, 18.5376797
1: -5.9060030, 14.9510040, -5.5205646, 14.2314167, -20.1374207, 20.4715691
2: -4.9519839, 16.7172890, -4.6168003, 15.9664316, -20.9184151, 21.3340874
3: -5.9551606, 21.4694099, -5.6251125, 20.5657425, -26.5209007, 27.0945168
4: -4.8511004, 19.7662888, -4.5940485, 18.9481487, -23.7992496, 24.3603363

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5170892, upper bound: 27.5025174
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5137558, upper bound: 27.5090164
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.0540481, 14.4764099, -3.8882565, 14.0178547, -18.0719032, 18.3646660
1: -5.8242021, 14.7517223, -5.5631576, 14.3240805, -20.1482811, 20.3148766
2: -4.8843389, 16.5075340, -4.6534100, 16.0694389, -20.9537754, 21.1609440
3: -5.8696055, 21.1784897, -5.6674542, 20.6948471, -26.5644512, 26.8459435
4: -4.7930698, 19.4997158, -4.6273518, 19.0684605, -23.8615303, 24.1270676

Time for backsubstitution: 2.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5310979, upper bound: 27.5151886
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5293449, upper bound: 27.5218874
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.1237140, 14.6792870, -4.4455428, 15.5328064, -19.6565189, 19.1248264
1: -5.9060030, 14.9510040, -6.3096666, 15.8786974, -21.7847004, 21.2606697
2: -4.9519839, 16.7172890, -5.3230810, 17.7998981, -22.7518787, 22.0403709
3: -5.9551606, 21.4694099, -6.3936768, 22.8474293, -28.8025894, 27.8630867
4: -4.8511004, 19.7662888, -5.2204366, 21.1061764, -25.9572754, 24.9867249

Time for backsubstitution: 2.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4936973, upper bound: 27.4793380
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5195109, upper bound: 27.5195934
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.0540481, 14.4764099, -4.4765372, 15.6225147, -19.6765614, 18.9529476
1: -5.8242021, 14.7517223, -6.3536415, 15.9724932, -21.7966957, 21.1053619
2: -4.8843389, 16.5075340, -5.3602648, 17.9039478, -22.7882862, 21.8677959
3: -5.8696055, 21.1784897, -6.4377475, 22.9774036, -28.8470058, 27.6162376
4: -4.7930698, 19.4997158, -5.2538066, 21.2279301, -26.0209999, 24.7535191

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5238816, upper bound: 27.5000860
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5238813, upper bound: 27.5221102
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.1237140, 14.6792870, -4.2099819, 15.0545835, -19.1782970, 18.8892689
1: -5.9060030, 14.9510040, -6.0341425, 15.3639421, -21.2699451, 20.9851456
2: -4.9519839, 16.7172890, -5.0320191, 17.1727810, -22.1247616, 21.7493076
3: -5.9551606, 21.4694099, -6.1294856, 22.1190701, -28.0742283, 27.5988960
4: -4.8511004, 19.7662888, -4.9741917, 20.3766880, -25.2277889, 24.7404804

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5231934, upper bound: 27.5199203
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5197310, upper bound: 27.5258829
time: 0.79 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 5.69 seconds
IS_A2_B2_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 5.69
Output dim: 3, lower bound: -27.5233782, upper bound: 27.5233782
IS_A2_B2_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.69
Output dim: 3, lower bound: -27.5233782, upper bound: 27.5322873
IS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 5.69
Output dim: 3, lower bound: -27.5057843, upper bound: 27.4873370
IS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 10, time: 5.69
Output dim: 3, lower bound: -27.4754364, upper bound: 27.4754364
IS_A2_B2_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 5.69
Output dim: 3, lower bound: -27.5128138, upper bound: 27.5202738
IS_A2_B2_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.69
Output dim: 3, lower bound: -27.5199470, upper bound: 27.5283393
IS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.69
Output dim: 3, lower bound: -27.5336110, upper bound: 27.5228199
IS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.69
Output dim: 3, lower bound: -27.5336110, upper bound: 27.5316258
IS_A2_B2_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.69
Output dim: 3, lower bound: -27.5216605, upper bound: 27.5173344
IS_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.69
Output dim: 3, lower bound: -27.5216605, upper bound: 27.5173344
IS_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 10, time: 5.69
Output dim: 3, lower bound: -27.5170892, upper bound: 27.5025174
IS_A2_B2_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 10, time: 5.69
Output dim: 3, lower bound: -27.5137558, upper bound: 27.5090164
IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.69
Output dim: 3, lower bound: -27.5310979, upper bound: 27.5151886
IS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.69
Output dim: 3, lower bound: -27.5293449, upper bound: 27.5218874
IS_A2_B2_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 5.69
Output dim: 3, lower bound: -27.4936973, upper bound: 27.4793380
IS_A2_B2_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 5.69
Output dim: 3, lower bound: -27.5195109, upper bound: 27.5195934
IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.69
Output dim: 3, lower bound: -27.5238816, upper bound: 27.5000860
IS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.69
Output dim: 3, lower bound: -27.5238813, upper bound: 27.5221102
IS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 5.69
Output dim: 3, lower bound: -27.5231934, upper bound: 27.5199203
IS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 5.69
Output dim: 3, lower bound: -27.5197310, upper bound: 27.5258829
IS_A2_B2_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.69
Output dim: 3, lower bound: -27.5342854, upper bound: 27.5329038
IS_A2_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.69
Output dim: 3, lower bound: -27.5248300, upper bound: 27.5322884
IS_A2_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.69
Output dim: 3, lower bound: -27.5342856, upper bound: 27.5329041
IS_A2_B2_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.69
Output dim: 3, lower bound: -27.5163825, upper bound: 27.5230258
IS_A2_B2_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.69
Output dim: 3, lower bound: -27.5206131, upper bound: 27.5206539
IS_A2_B2_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.69
Output dim: 3, lower bound: -27.5163825, upper bound: 27.5230364
IS_A2_B2_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.69
Output dim: 3, lower bound: -27.5163825, upper bound: 27.5207305
IS_A2_B2_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.69
Output dim: 3, lower bound: -27.5175703, upper bound: 27.5251850
IS_A2_B2_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.69
Output dim: 3, lower bound: -27.5233213, upper bound: 27.5261421
IS_A2_B2_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.69
Output dim: 3, lower bound: -27.5179896, upper bound: 27.5303457
IS_A2_B2_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.69
Output dim: 3, lower bound: -27.5222745, upper bound: 27.5278047
IS_A2_B2_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.69
Output dim: 3, lower bound: -27.5179897, upper bound: 27.5317299
IS_A2_B2_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.69
Output dim: 3, lower bound: -27.5222744, upper bound: 27.5278045
IS_A2_B2_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.69
Output dim: 3, lower bound: -27.5184059, upper bound: 27.5314126
IS_A2_B2_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.69
Output dim: 3, lower bound: -27.5249386, upper bound: 27.5333894
IS_A2_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.69
Output dim: 3, lower bound: -27.5184058, upper bound: 27.5325318
IS_A2_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.69
Output dim: 3, lower bound: -27.5249386, upper bound: 27.5345394
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.69
Output dim: 3, lower bound: -27.5341136, upper bound: 27.5278002
IS_A2_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.69
Output dim: 3, lower bound: -27.5223967, upper bound: 27.5204887
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=31.572500228881836
rel_dist={3: [-27.545485351818588, 27.545485351818584]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5233805, upper bound: 27.5335250
time: 0.73 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5401791, upper bound: 27.5401793
time: 0.80 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.74 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.74
Output dim: 3, lower bound: -27.5233805, upper bound: 27.5335250
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.74
Output dim: 3, lower bound: -27.5401791, upper bound: 27.5401793

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.4779983, 15.6032467, -4.7866821, 16.3863354, -20.8643341, 20.3899288
1: -6.4072104, 15.9771137, -6.8432279, 16.8213444, -23.2285538, 22.8203411
2: -5.4055552, 17.9303551, -5.7925940, 18.8647137, -24.2702694, 23.7229500
3: -6.4871049, 22.9893456, -6.9157748, 24.1210041, -30.6081085, 29.9051208
4: -5.3025470, 21.2900162, -5.6431189, 22.3685188, -27.6710663, 26.9331360

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5196270, upper bound: 27.5318398
time: 0.79 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5233583, upper bound: 27.5335250
time: 0.71 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -4.8002801, 16.4875641, -4.8284826, 16.4942989, -21.2945786, 21.3160477
1: -6.8583674, 16.9091358, -6.9039440, 16.9393597, -23.7977276, 23.8130741
2: -5.7989798, 18.9445438, -5.8449163, 18.9991856, -24.7981644, 24.7894573
3: -6.9374380, 24.2584801, -6.9747157, 24.2811546, -31.2185917, 31.2331963
4: -5.6505985, 22.4603367, -5.6895337, 22.5205956, -28.1711922, 28.1498699

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5367800, upper bound: 27.5381645
time: 0.84 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5400965, upper bound: 27.5400967
time: 0.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.13 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 3, lower bound: -27.5196270, upper bound: 27.5318398
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 3, lower bound: -27.5233583, upper bound: 27.5335250
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 3, lower bound: -27.5367800, upper bound: 27.5381645
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 3, lower bound: -27.5400965, upper bound: 27.5400967

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -4.1382656, 14.5957584, -4.0778384, 14.2979774, -18.4362431, 18.6735973
1: -5.9143834, 14.9243336, -5.9258318, 14.6709547, -20.5853367, 20.8501625
2: -4.9759965, 16.7641430, -5.0128708, 16.5077305, -21.4837265, 21.7770138
3: -5.9773612, 21.5092087, -5.9401674, 21.0567207, -27.0340805, 27.4493752
4: -4.9085326, 19.9155521, -4.9011049, 19.5171490, -24.4256821, 24.8166580

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5107854, upper bound: 27.5261744
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4988357, upper bound: 27.5177518
time: 0.79 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -4.4418936, 15.4881372, -4.7045069, 16.1266613, -20.5685482, 20.1926441
1: -6.3562999, 15.8623877, -6.7274027, 16.5618343, -22.9181347, 22.5897884
2: -5.3623009, 17.8052959, -5.6940155, 18.5832672, -23.9455681, 23.4993114
3: -6.4357409, 22.8261089, -6.7989855, 23.7537174, -30.1894588, 29.6250954
4: -5.2633905, 21.1458282, -5.5542841, 22.0420914, -27.3054810, 26.7001114

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5199642, upper bound: 27.5295046
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5199642, upper bound: 27.5335250
time: 0.79 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -4.4560103, 15.4839010, -4.0987930, 14.3596191, -18.8156281, 19.5826912
1: -6.3589268, 15.8586674, -5.9590111, 14.7385178, -21.0974426, 21.8176785
2: -5.3669195, 17.7763519, -5.0401125, 16.5851898, -21.9521084, 22.8164616
3: -6.4302497, 22.7891922, -5.9721651, 21.1508007, -27.5810490, 28.7613564
4: -5.2585702, 21.0940018, -4.9262757, 19.6014023, -24.8599701, 26.0202770

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5235521, upper bound: 27.5333215
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5348490, upper bound: 27.5367055
time: 0.81 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -4.7636456, 16.3704605, -4.7439461, 16.2283802, -20.9920254, 21.1144028
1: -6.8063335, 16.7921143, -6.7844915, 16.6728058, -23.4791374, 23.5766068
2: -5.7550316, 18.8174706, -5.7433257, 18.7096748, -24.4647064, 24.5607967
3: -6.8844986, 24.0920811, -6.8542533, 23.9047222, -30.7892208, 30.9463310
4: -5.6106577, 22.3131275, -5.5979695, 22.1849537, -27.7956123, 27.9110966

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5278504, upper bound: 27.5348744
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5379820, upper bound: 27.5379824
time: 0.68 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.24 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 3, lower bound: -27.5107854, upper bound: 27.5261744
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.24
Output dim: 3, lower bound: -27.4988357, upper bound: 27.5177518
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 3, lower bound: -27.5199642, upper bound: 27.5295046
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 3, lower bound: -27.5199642, upper bound: 27.5335250
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 3, lower bound: -27.5235521, upper bound: 27.5333215
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 3, lower bound: -27.5348490, upper bound: 27.5367055
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 3, lower bound: -27.5278504, upper bound: 27.5348744
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 3, lower bound: -27.5379820, upper bound: 27.5379824

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.8845038, 13.9131289, -4.0778384, 14.2979774, -18.1824780, 17.9909630
1: -5.5398741, 14.1960001, -5.9258318, 14.6709547, -20.2108288, 20.1218300
2: -4.6540227, 15.9607315, -5.0128708, 16.5077305, -21.1617508, 20.9736023
3: -5.6175628, 20.5132713, -5.9401674, 21.0567207, -26.6742821, 26.4534378
4: -4.6185470, 18.9634056, -4.9011049, 19.5171490, -24.1356945, 23.8645096

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4857923, upper bound: 27.4894564
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4857923, upper bound: 27.4894564
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.8096104, 13.5872917, -4.7045069, 16.1266613, -19.9362698, 18.2917976
1: -5.5293055, 13.9026499, -6.7274027, 16.5618343, -22.0911407, 20.6300526
2: -4.6612096, 15.6515274, -5.6940155, 18.5832672, -23.2444763, 21.3455429
3: -5.5544548, 20.0211716, -6.7989855, 23.7537174, -29.3081722, 26.8201561
4: -4.5881572, 18.5381012, -5.5542841, 22.0420914, -26.6302471, 24.0923843

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5120602, upper bound: 27.5136508
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5120602, upper bound: 27.5295046
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.3976188, 15.3468742, -4.7045069, 16.1266613, -20.5242767, 20.0513802
1: -6.2940626, 15.7216339, -6.7274027, 16.5618343, -22.8558960, 22.4490356
2: -5.3095002, 17.6520042, -5.6940155, 18.5832672, -23.8927670, 23.3460197
3: -6.3730483, 22.6263657, -6.7989855, 23.7537174, -30.1267662, 29.4253502
4: -5.2157865, 20.9694633, -5.5542841, 22.0420914, -27.2578773, 26.5237465

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5120602, upper bound: 27.5120602
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5120602, upper bound: 27.5120602
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.1883216, 14.7835121, -4.0987930, 14.3596191, -18.5479412, 18.8823032
1: -5.9703255, 15.1122866, -5.9590111, 14.7385178, -20.7088432, 21.0712967
2: -5.0377016, 16.9457092, -5.0401125, 16.5851898, -21.6228905, 21.9858189
3: -6.0523725, 21.7681942, -5.9721651, 21.1508007, -27.2031727, 27.7403603
4: -4.9621649, 20.1095657, -4.9262757, 19.6014023, -24.5635681, 25.0358410

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4923523, upper bound: 27.4937042
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4923524, upper bound: 27.5272879
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.5449395, 15.9130096, -4.0145268, 14.1432114, -18.6881504, 19.9275341
1: -6.4959974, 16.2446594, -5.8413587, 14.5076876, -21.0036850, 22.0860176
2: -5.4457636, 18.1547642, -4.9387350, 16.3245373, -21.7703018, 23.0934982
3: -6.5620146, 23.3148518, -5.8588223, 20.8320293, -27.3940430, 29.1736698
4: -5.3320489, 21.5337486, -4.8346682, 19.2966843, -24.6287327, 26.3684158

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5161947, upper bound: 27.5006528
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5161949, upper bound: 27.5362773
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.4805202, 15.6340685, -4.7439461, 16.2283802, -20.7089005, 20.3780136
1: -6.4036279, 16.0067654, -6.7844915, 16.6728058, -23.0764332, 22.7912560
2: -5.4078512, 17.9396744, -5.7433257, 18.7096748, -24.1175270, 23.6829987
3: -6.4891987, 23.0174026, -6.8542533, 23.9047222, -30.3939209, 29.8716526
4: -5.2989187, 21.2770100, -5.5979695, 22.1849537, -27.4838715, 26.8749790

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4965856, upper bound: 27.4948762
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4965858, upper bound: 27.5308079
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.8176165, 16.7194538, -4.6507306, 15.9857750, -20.8033905, 21.3701839
1: -6.8903656, 17.0972080, -6.6552248, 16.4143391, -23.3047047, 23.7524300
2: -5.7917786, 19.0957966, -5.6309738, 18.4174938, -24.2092724, 24.7267666
3: -6.9725718, 24.5095921, -6.7252245, 23.5475311, -30.5200996, 31.2348156
4: -5.6507530, 22.6451244, -5.4967818, 21.8438644, -27.4946175, 28.1419029

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5184556, upper bound: 27.5010887
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5184558, upper bound: 27.5379561
time: 0.92 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.23 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 3, lower bound: -27.4857923, upper bound: 27.4894564
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 3, lower bound: -27.4857923, upper bound: 27.4894564
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 3, lower bound: -27.5120602, upper bound: 27.5136508
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 3, lower bound: -27.5120602, upper bound: 27.5295046
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 3, lower bound: -27.5120602, upper bound: 27.5120602
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 3, lower bound: -27.5120602, upper bound: 27.5120602
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 3, lower bound: -27.4923523, upper bound: 27.4937042
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 3, lower bound: -27.4923524, upper bound: 27.5272879
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 3, lower bound: -27.5161947, upper bound: 27.5006528
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 3, lower bound: -27.5161949, upper bound: 27.5362773
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 3, lower bound: -27.4965856, upper bound: 27.4948762
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 3, lower bound: -27.4965858, upper bound: 27.5308079
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.23
Output dim: 3, lower bound: -27.5184556, upper bound: 27.5010887
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 3, lower bound: -27.5184558, upper bound: 27.5379561

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.8096104, 13.5872917, -4.7206116, 16.2323303, -20.0419388, 18.3079014
1: -5.5293055, 13.9026499, -6.7455416, 16.6543579, -22.1836624, 20.6481915
2: -4.6612096, 15.6515274, -5.7037611, 18.6685505, -23.3297596, 21.3552837
3: -5.5544548, 20.0211716, -6.8226271, 23.8968391, -29.4512939, 26.8437996
4: -4.5881572, 18.5381012, -5.5644245, 22.1408634, -26.7290192, 24.1025238

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4849823, upper bound: 27.4872111
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4804599, upper bound: 27.4807055
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.1883216, 14.7835121, -4.0815225, 14.3603363, -18.5486584, 18.8650341
1: -5.9703255, 15.1122866, -5.9201736, 14.7163811, -20.6867065, 21.0324593
2: -5.0377016, 16.9457092, -5.0029421, 16.5449162, -21.5826187, 21.9486504
3: -6.0523725, 21.7681942, -5.9411888, 21.1416397, -27.1940117, 27.7093811
4: -4.9621649, 20.1095657, -4.8949690, 19.5565891, -24.5187531, 25.0045357

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4922621, upper bound: 27.4914540
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4922621, upper bound: 27.5181589
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.5449395, 15.9130096, -3.9896507, 14.1232405, -18.6681805, 19.9026585
1: -6.4959974, 16.2446594, -5.7917428, 14.4631653, -20.9591637, 22.0364017
2: -5.4457636, 18.1547642, -4.8917294, 16.2582150, -21.7039795, 23.0464935
3: -6.5620146, 23.3148518, -5.8171506, 20.7924175, -27.3544312, 29.1320019
4: -5.3320489, 21.5337486, -4.7946362, 19.2213078, -24.5533562, 26.3283844

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5160712, upper bound: 27.5342286
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5160712, upper bound: 27.5362773
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.4805202, 15.6340685, -4.7206116, 16.2323303, -20.7128448, 20.3546772
1: -6.4036279, 16.0067654, -6.7455416, 16.6543579, -23.0579853, 22.7523079
2: -5.4078512, 17.9396744, -5.7037611, 18.6685505, -24.0764008, 23.6434307
3: -6.4891987, 23.0174026, -6.8226271, 23.8968391, -30.3860378, 29.8400288
4: -5.2989187, 21.2770100, -5.5644245, 22.1408634, -27.4397812, 26.8414326

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4922621, upper bound: 27.5188748
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4922621, upper bound: 27.4914540
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.8176165, 16.7194538, -4.6183882, 15.9683008, -20.7859173, 21.3378410
1: -6.8903656, 17.0972080, -6.6031342, 16.3727589, -23.2631245, 23.7003403
2: -5.7917786, 19.0957966, -5.5814114, 18.3495636, -24.1413422, 24.6772041
3: -6.9725718, 24.5095921, -6.6841116, 23.5074482, -30.4800186, 31.1937027
4: -5.6507530, 22.6451244, -5.4543152, 21.7686558, -27.4194088, 28.0994396

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5160712, upper bound: 27.5348493
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5160712, upper bound: 27.5010887
time: 0.90 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.24 seconds
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.4849823, upper bound: 27.4872111
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.4804599, upper bound: 27.4807055
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.4922621, upper bound: 27.4914540
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.4922621, upper bound: 27.5181589
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5160712, upper bound: 27.5342286
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5160712, upper bound: 27.5362773
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.4922621, upper bound: 27.5188748
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.4922621, upper bound: 27.4914540
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5160712, upper bound: 27.5348493
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5160712, upper bound: 27.5010887

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.1616135, 14.7958717, -3.9896507, 14.1232405, -18.2848492, 18.7855225
1: -6.0217419, 15.1112652, -5.7917428, 14.4631653, -20.4849052, 20.9030075
2: -5.0678825, 16.9106216, -4.8917294, 16.2582150, -21.3260975, 21.8023510
3: -6.0606642, 21.6794033, -5.8171506, 20.7924175, -26.8530807, 27.4965534
4: -4.9605665, 20.0012703, -4.7946362, 19.2213078, -24.1818733, 24.7959061

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5292357, upper bound: 27.5224278
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5292360, upper bound: 27.5338232
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.7674384, 16.5644913, -3.9896507, 14.1232405, -18.8906765, 20.5541401
1: -6.8205609, 16.9415417, -5.7917428, 14.4631653, -21.2837219, 22.7332840
2: -5.7317390, 18.9258461, -4.8917294, 16.2582150, -21.9899540, 23.8175755
3: -6.9023232, 24.2892647, -5.8171506, 20.7924175, -27.6947403, 30.1064148
4: -5.5971479, 22.4497318, -4.7946362, 19.2213078, -24.8184547, 27.2443676

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5292357, upper bound: 27.5224278
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5292360, upper bound: 27.5224278
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.1616135, 14.7958717, -4.6183882, 15.9683008, -20.1299095, 19.4142609
1: -6.0217419, 15.1112652, -6.6031342, 16.3727589, -22.3944988, 21.7143974
2: -5.0678825, 16.9106216, -5.5814114, 18.3495636, -23.4174461, 22.4920311
3: -6.0606642, 21.6794033, -6.6841116, 23.5074482, -29.5681114, 28.3635139
4: -4.9605665, 20.0012703, -5.4543152, 21.7686558, -26.7292213, 25.4555855

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5292357, upper bound: 27.5235521
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5292360, upper bound: 27.5235521
time: 0.93 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.58 seconds
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 3, lower bound: -27.5292357, upper bound: 27.5224278
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 3, lower bound: -27.5292360, upper bound: 27.5338232
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 3, lower bound: -27.5292357, upper bound: 27.5224278
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 3, lower bound: -27.5292360, upper bound: 27.5224278
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 3, lower bound: -27.5292357, upper bound: 27.5235521
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 3, lower bound: -27.5292360, upper bound: 27.5235521

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.1616135, 14.7958717, -3.8377326, 13.7042408, -17.8658524, 18.6336040
1: -6.0217419, 15.1112652, -5.5586572, 14.0156584, -20.0373993, 20.6699181
2: -5.0678825, 16.9106216, -4.6897707, 15.7574348, -20.8253174, 21.6003914
3: -6.0606642, 21.6794033, -5.5885458, 20.1776047, -26.2382660, 27.2679482
4: -4.9605665, 20.0012703, -4.6097212, 18.6277122, -23.5882797, 24.6109924

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5176709, upper bound: 27.5001271
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5162549, upper bound: 27.4999665
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.1616135, 14.7958717, -4.1616135, 14.7958717, -18.9574833, 18.9574833
1: -6.0217419, 15.1112652, -6.0217419, 15.1112652, -21.1330032, 21.1330032
2: -5.0678825, 16.9106216, -5.0678825, 16.9106216, -21.9785042, 21.9785042
3: -6.0606642, 21.6794033, -6.0606642, 21.6794033, -27.7400665, 27.7400665
4: -4.9605665, 20.0012703, -4.9605665, 20.0012703, -24.9618378, 24.9618378

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5176711, upper bound: 27.5001271
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5162555, upper bound: 27.4999666
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.7674384, 16.5644913, -3.8377326, 13.7042408, -18.4716797, 20.4022217
1: -6.8205609, 16.9415417, -5.5586572, 14.0156584, -20.8362198, 22.5001965
2: -5.7317390, 18.9258461, -4.6897707, 15.7574348, -21.4891739, 23.6156139
3: -6.9023232, 24.2892647, -5.5885458, 20.1776047, -27.0799274, 29.8778114
4: -5.5971479, 22.4497318, -4.6097212, 18.6277122, -24.2248611, 27.0594521

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5177764, upper bound: 27.5010777
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5187321, upper bound: 27.5027081
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.7674384, 16.5644913, -4.1616135, 14.7958717, -19.5633106, 20.7260990
1: -6.8205609, 16.9415417, -6.0217419, 15.1112652, -21.9318199, 22.9632816
2: -5.7317390, 18.9258461, -5.0678825, 16.9106216, -22.6423607, 23.9937286
3: -6.9023232, 24.2892647, -6.0606642, 21.6794033, -28.5817261, 30.3499279
4: -5.5971479, 22.4497318, -4.9605665, 20.0012703, -25.5984173, 27.4102974

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5177766, upper bound: 27.5297304
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5187325, upper bound: 27.5027082
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.1616135, 14.7958717, -4.4364772, 15.4927216, -19.6543350, 19.2323494
1: -6.0217419, 15.1112652, -6.3414073, 15.8659678, -21.8877106, 21.4526691
2: -5.0678825, 16.9106216, -5.3552303, 17.7862968, -22.8541794, 22.2658520
3: -6.0606642, 21.6794033, -6.4260044, 22.8168812, -28.8775444, 28.1054077
4: -4.9605665, 20.0012703, -5.2513251, 21.1006165, -26.0611839, 25.2525940

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5310793, upper bound: 27.5197075
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5268513, upper bound: 27.5124450
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.1616135, 14.7958717, -4.7756276, 16.5842285, -20.7458382, 19.5714989
1: -6.0217419, 15.1112652, -6.8313746, 16.9627132, -22.9844551, 21.9426384
2: -5.0678825, 16.9106216, -5.7417583, 18.9490433, -24.0169258, 22.6523781
3: -6.0606642, 21.6794033, -6.9129910, 24.3177433, -30.3784065, 28.5923901
4: -4.9605665, 20.0012703, -5.6055784, 22.4766064, -27.4371719, 25.6068459

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5310795, upper bound: 27.5327622
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5162554, upper bound: 27.4999666
time: 0.86 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.48 seconds
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 3, lower bound: -27.5176709, upper bound: 27.5001271
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 3, lower bound: -27.5162549, upper bound: 27.4999665
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 3, lower bound: -27.5176711, upper bound: 27.5001271
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 3, lower bound: -27.5162555, upper bound: 27.4999666
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 3, lower bound: -27.5177764, upper bound: 27.5010777
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 3, lower bound: -27.5187321, upper bound: 27.5027081
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.48
Output dim: 3, lower bound: -27.5177766, upper bound: 27.5297304
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 3, lower bound: -27.5187325, upper bound: 27.5027082
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.48
Output dim: 3, lower bound: -27.5310793, upper bound: 27.5197075
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.48
Output dim: 3, lower bound: -27.5268513, upper bound: 27.5124450
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.48
Output dim: 3, lower bound: -27.5310795, upper bound: 27.5327622
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.48
Output dim: 3, lower bound: -27.5162554, upper bound: 27.4999666

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.2393880, 15.1400146, -3.9243522, 14.1257553, -18.3651428, 19.0643654
1: -6.0753212, 15.4536896, -5.6636448, 14.4075222, -20.4828415, 21.1173306
2: -5.0675325, 17.2718410, -4.7476602, 16.1241207, -21.1916542, 22.0195007
3: -6.1711316, 22.2439194, -5.7086349, 20.7131214, -26.8842506, 27.9525528
4: -5.0065122, 20.4930534, -4.6697898, 19.0728893, -24.0793991, 25.1628399

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5314564, upper bound: 27.5297299
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5314564, upper bound: 27.5297299
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -4.1801624, 14.8041792, -18.4506340, 17.5303516
1: -5.2464833, 13.5860958, -5.9743381, 15.1460457, -20.3925266, 19.5604343
2: -4.3717928, 15.2154961, -5.0205998, 16.9822750, -21.3540688, 20.2360954
3: -5.2993393, 19.6015205, -6.0706930, 21.8144665, -27.1138058, 25.6722126
4: -4.3268204, 18.0051994, -4.9499264, 20.1450100, -24.4718304, 22.9551239

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5247626, upper bound: 27.5114365
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5247626, upper bound: 27.5124450
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.1191621, 14.6685381, -4.3800497, 15.3396292, -19.4587917, 19.0485878
1: -5.9215693, 14.9546156, -6.2653298, 15.7050419, -21.6266117, 21.2199459
2: -4.9671373, 16.7323151, -5.2894130, 17.6113205, -22.5784569, 22.0217285
3: -5.9684920, 21.4562817, -6.3493214, 22.6010933, -28.5695801, 27.8056011
4: -4.8678341, 19.7656593, -5.1933098, 20.8934021, -25.7612362, 24.9589691

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5247626, upper bound: 27.5114365
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5247626, upper bound: 27.5114365
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -4.5286918, 15.9218464, -19.5683041, 17.8788834
1: -5.2464833, 13.5860958, -6.4815860, 16.2707996, -21.5172787, 20.0676823
2: -4.3717928, 15.2154961, -5.4201164, 18.1784019, -22.5501938, 20.6356125
3: -5.2993393, 19.6015205, -6.5724845, 23.3582268, -28.6575661, 26.1740036
4: -4.3268204, 18.0051994, -5.3183122, 21.5639420, -25.8907623, 23.3235073

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5310912, upper bound: 27.5304677
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5310912, upper bound: 27.5322130
time: 1.04 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.55 seconds
IS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 3, lower bound: -27.5314564, upper bound: 27.5297299
IS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 3, lower bound: -27.5314564, upper bound: 27.5297299
IS_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 3, lower bound: -27.5247626, upper bound: 27.5114365
IS_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 3, lower bound: -27.5247626, upper bound: 27.5124450
IS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 3, lower bound: -27.5247626, upper bound: 27.5114365
IS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 3, lower bound: -27.5247626, upper bound: 27.5114365
IS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 3, lower bound: -27.5310912, upper bound: 27.5304677
IS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.55
Output dim: 3, lower bound: -27.5310912, upper bound: 27.5322130

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.2393880, 15.1400146, -3.6464579, 13.3501921, -17.5895786, 18.7864723
1: -6.0753212, 15.4536896, -5.2464833, 13.5860958, -19.6614170, 20.7001705
2: -5.0675325, 17.2718410, -4.3717928, 15.2154961, -20.2830276, 21.6436348
3: -6.1711316, 22.2439194, -5.2993393, 19.6015205, -25.7726517, 27.5432587
4: -5.0065122, 20.4930534, -4.3268204, 18.0051994, -23.0117073, 24.8198719

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5278057, upper bound: 27.5297009
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5314548, upper bound: 27.5297285
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.2393880, 15.1400146, -4.1191621, 14.6685381, -18.9079266, 19.2591763
1: -6.0753212, 15.4536896, -5.9215693, 14.9546156, -21.0299377, 21.3752537
2: -5.0675325, 17.2718410, -4.9671373, 16.7323151, -21.7998466, 22.2389793
3: -6.1711316, 22.2439194, -5.9684920, 21.4562817, -27.6274109, 28.2124081
4: -5.0065122, 20.4930534, -4.8678341, 19.7656593, -24.7721710, 25.3608875

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5278057, upper bound: 27.5297009
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5314548, upper bound: 27.5297285
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -3.8882565, 14.0178547, -17.6643124, 17.2384453
1: -5.2464833, 13.5860958, -5.5631576, 14.3240805, -19.5705624, 19.1492538
2: -4.3717928, 15.2154961, -4.6534100, 16.0694389, -20.4412289, 19.8689060
3: -5.2993393, 19.6015205, -5.6674542, 20.6948471, -25.9941864, 25.2689743
4: -4.3268204, 18.0051994, -4.6273518, 19.0684605, -23.3952808, 22.6325512

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5191327, upper bound: 27.5111406
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5297813, upper bound: 27.5195203
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -4.4765372, 15.6225147, -19.2689705, 17.8267288
1: -5.2464833, 13.5860958, -6.3536415, 15.9724932, -21.2189770, 19.9397373
2: -4.3717928, 15.2154961, -5.3602648, 17.9039478, -22.2757397, 20.5757599
3: -5.2993393, 19.6015205, -6.4377475, 22.9774036, -28.2767410, 26.0392685
4: -4.3268204, 18.0051994, -5.2538066, 21.2279301, -25.5547504, 23.2590027

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5191327, upper bound: 27.5111406
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5297812, upper bound: 27.5195203
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.1191621, 14.6685381, -3.8882565, 14.0178547, -18.1370163, 18.5567932
1: -5.9215693, 14.9546156, -5.5631576, 14.3240805, -20.2456493, 20.5177727
2: -4.9671373, 16.7323151, -4.6534100, 16.0694389, -21.0365753, 21.3857250
3: -5.9684920, 21.4562817, -5.6674542, 20.6948471, -26.6633358, 27.1237354
4: -4.8678341, 19.7656593, -4.6273518, 19.0684605, -23.9362946, 24.3930111

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5175722, upper bound: 27.5087505
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5245097, upper bound: 27.5113944
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.1191621, 14.6685381, -4.4765372, 15.6225147, -19.7416763, 19.1450748
1: -5.9215693, 14.9546156, -6.3536415, 15.9724932, -21.8940620, 21.3082581
2: -4.9671373, 16.7323151, -5.3602648, 17.9039478, -22.8710861, 22.0925789
3: -5.9684920, 21.4562817, -6.4377475, 22.9774036, -28.9458904, 27.8940277
4: -4.8678341, 19.7656593, -5.2538066, 21.2279301, -26.0957642, 25.0194645

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5175723, upper bound: 27.5087506
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5245097, upper bound: 27.5113944
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -4.2396750, 15.1407347, -18.7871933, 17.5898647
1: -5.2464833, 13.5860958, -6.0757089, 15.4544592, -20.7009392, 19.6618042
2: -4.3717928, 15.2154961, -5.0678725, 17.2726898, -21.6444817, 20.2833691
3: -5.2993393, 19.6015205, -6.1715178, 22.2449474, -27.5442867, 25.7730389
4: -4.3268204, 18.0051994, -5.0068088, 20.4940319, -24.8208523, 23.0120087

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5242808, upper bound: 27.5284770
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5328290, upper bound: 27.5314200
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.6464579, 13.3501921, -4.8151884, 16.7127552, -20.3592129, 18.1653767
1: -5.2464833, 13.5860958, -6.8487349, 17.0677147, -22.3141975, 20.4348297
2: -4.3717928, 15.2154961, -5.7475977, 19.0672207, -23.4390125, 20.9630890
3: -5.2993393, 19.6015205, -6.9301233, 24.4769001, -29.7762394, 26.5316429
4: -4.3268204, 18.0051994, -5.6081338, 22.6030693, -26.9298878, 23.6133327

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5242808, upper bound: 27.5295725
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5328290, upper bound: 27.5314201
time: 0.85 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 5.02 seconds
IS_A2_B1_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 3, lower bound: -27.5278057, upper bound: 27.5297009
IS_A2_B1_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 3, lower bound: -27.5314548, upper bound: 27.5297285
IS_A2_B1_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 3, lower bound: -27.5278057, upper bound: 27.5297009
IS_A2_B1_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 3, lower bound: -27.5314548, upper bound: 27.5297285
IS_A2_B2_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 5.02
Output dim: 3, lower bound: -27.5191327, upper bound: 27.5111406
IS_A2_B2_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 3, lower bound: -27.5297813, upper bound: 27.5195203
IS_A2_B2_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.02
Output dim: 3, lower bound: -27.5191327, upper bound: 27.5111406
IS_A2_B2_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 3, lower bound: -27.5297812, upper bound: 27.5195203
IS_A2_B2_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 5.02
Output dim: 3, lower bound: -27.5175722, upper bound: 27.5087505
IS_A2_B2_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 3, lower bound: -27.5245097, upper bound: 27.5113944
IS_A2_B2_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.02
Output dim: 3, lower bound: -27.5175723, upper bound: 27.5087506
IS_A2_B2_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 3, lower bound: -27.5245097, upper bound: 27.5113944
IS_A2_B2_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 3, lower bound: -27.5242808, upper bound: 27.5284770
IS_A2_B2_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 3, lower bound: -27.5328290, upper bound: 27.5314200
IS_A2_B2_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 3, lower bound: -27.5242808, upper bound: 27.5295725
IS_A2_B2_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 3, lower bound: -27.5328290, upper bound: 27.5314201

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.1652908, 14.8957834, -3.6405568, 13.3329592, -17.4982491, 18.5363407
1: -5.9683876, 15.2039032, -5.2379394, 13.5679045, -19.5362911, 20.4418373
2: -4.9834766, 16.9886322, -4.3645067, 15.1954508, -20.1789246, 21.3531380
3: -6.0599270, 21.8858299, -5.2906837, 19.5766811, -25.6366081, 27.1765118
4: -4.9279890, 20.1602840, -4.3202229, 17.9813309, -22.9093189, 24.4805069

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4408190, upper bound: 27.4518655
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4337297, upper bound: 27.4489984
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.1891284, 15.0038404, -3.6374068, 13.3253632, -17.5144882, 18.6412449
1: -6.0037899, 15.3114023, -5.2332439, 13.5598269, -19.5636139, 20.5446472
2: -5.0072761, 17.1143532, -4.3602142, 15.1866503, -20.1939240, 21.4745674
3: -6.1000462, 22.0504875, -5.2859416, 19.5661736, -25.6662197, 27.3364296
4: -4.9527273, 20.3066311, -4.3164282, 17.9710445, -22.9237709, 24.6230583

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4949767, upper bound: 27.4984165
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4756057, upper bound: 27.4805101
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.1652908, 14.8957834, -4.1136179, 14.6518850, -18.8171749, 19.0094013
1: -5.9683876, 15.2039032, -5.9134521, 14.9371271, -20.9055119, 21.1173534
2: -4.9834766, 16.9886322, -4.9602675, 16.7129898, -21.6964664, 21.9489002
3: -6.0599270, 21.8858299, -5.9602323, 21.4323101, -27.4922371, 27.8460617
4: -4.9279890, 20.1602840, -4.8616056, 19.7426853, -24.6706734, 25.0218887

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4857993, upper bound: 27.4911214
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5275780, upper bound: 27.5294906
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.1891284, 15.0038404, -4.1098576, 14.6428070, -18.8319321, 19.1136971
1: -6.0037899, 15.3114023, -5.9078822, 14.9275293, -20.9313183, 21.2192841
2: -5.0072761, 17.1143532, -4.9552546, 16.7024117, -21.7096882, 22.0696049
3: -6.1000462, 22.0504875, -5.9546351, 21.4196644, -27.5197105, 28.0051231
4: -4.9527273, 20.3066311, -4.8572178, 19.7302608, -24.6829872, 25.1638470

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4828474, upper bound: 27.4816232
time: 1.32 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5313305, upper bound: 27.5295223
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -3.5840602, 13.1680984, -3.8711884, 13.9692421, -17.5533028, 17.0392876
1: -5.1535125, 13.3938541, -5.5387783, 14.2728987, -19.4264088, 18.9326305
2: -4.2921443, 15.0030842, -4.6324234, 16.0127163, -20.3048611, 19.6355076
3: -5.2045627, 19.3389301, -5.6431541, 20.6246128, -25.8291759, 24.9820843
4: -4.2549839, 17.7536049, -4.6085286, 19.0013771, -23.2563610, 22.3621292

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294437, upper bound: 27.5161280
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5291661, upper bound: 27.5201675
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.5840602, 13.1680984, -4.4578252, 15.5703077, -19.1543674, 17.6259232
1: -5.1535125, 13.3938541, -6.3264751, 15.9174957, -21.0710068, 19.7203274
2: -4.2921443, 15.0030842, -5.3373003, 17.8429699, -22.1351128, 20.3403854
3: -5.2045627, 19.3389301, -6.4112501, 22.9013214, -28.1058846, 25.7501793
4: -4.2549839, 17.7536049, -5.2333546, 21.1556053, -25.4105873, 22.9869595

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5052457, upper bound: 27.4829690
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5304591, upper bound: 27.5189092
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.0540481, 14.4764099, -3.8711884, 13.9692421, -18.0232906, 18.3475990
1: -5.8242021, 14.7517223, -5.5387783, 14.2728987, -20.0970993, 20.2904987
2: -4.8843389, 16.5075340, -4.6324234, 16.0127163, -20.8970547, 21.1399536
3: -5.8696055, 21.1784897, -5.6431541, 20.6246128, -26.4942169, 26.8216438
4: -4.7930698, 19.4997158, -4.6085286, 19.0013771, -23.7944469, 24.1082401

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5245080, upper bound: 27.5068773
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5245080, upper bound: 27.5115666
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.0540481, 14.4764099, -4.4578252, 15.5703077, -19.6243553, 18.9342346
1: -5.8242021, 14.7517223, -6.3264751, 15.9174957, -21.7416973, 21.0781937
2: -4.8843389, 16.5075340, -5.3373003, 17.8429699, -22.7273064, 21.8448334
3: -5.8696055, 21.1784897, -6.4112501, 22.9013214, -28.7709255, 27.5897408
4: -4.7930698, 19.4997158, -5.2333546, 21.1556053, -25.9486752, 24.7330704

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4994839, upper bound: 27.4788255
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5222219, upper bound: 27.5092144
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.6164749, 13.2819958, -4.1723442, 14.9448881, -18.5613594, 17.4543381
1: -5.1846161, 13.4975052, -5.9813809, 15.2487183, -20.4333344, 19.4788837
2: -4.3129835, 15.1071482, -4.9865980, 17.0455894, -21.3585739, 20.0937424
3: -5.2438045, 19.4920826, -6.0761089, 21.9587116, -27.2025166, 25.5681896
4: -4.2717013, 17.8941593, -4.9327221, 20.2272320, -24.4989338, 22.8268814

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5237602, upper bound: 27.5235139
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5237602, upper bound: 27.5285237
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -3.5840602, 13.1680984, -4.2231445, 15.0941725, -18.6782322, 17.3912430
1: -5.1535125, 13.3938541, -6.0520325, 15.4053993, -20.5589104, 19.4458847
2: -4.2921443, 15.0030842, -5.0477176, 17.2183247, -21.5104675, 20.0508022
3: -5.2045627, 19.3389301, -6.1479211, 22.1772938, -27.3818550, 25.4868507
4: -4.2549839, 17.7536049, -4.9886756, 20.4294395, -24.6844234, 22.7422810

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4987126, upper bound: 27.4851986
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4884721, upper bound: 27.4807831
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.6164749, 13.2819958, -4.7444897, 16.5072422, -20.1237125, 18.0264797
1: -5.1846161, 13.4975052, -6.7493005, 16.8519287, -22.0365448, 20.2468052
2: -4.3129835, 15.1071482, -5.6626158, 18.8287811, -23.1417637, 20.7697620
3: -5.2438045, 19.4920826, -6.8290033, 24.1776237, -29.4214287, 26.3210850
4: -4.2717013, 17.8941593, -5.5311451, 22.3230267, -26.5947285, 23.4253044

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5125649, upper bound: 27.5165320
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5240240, upper bound: 27.5280145
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.5840602, 13.1680984, -4.7961559, 16.6599960, -20.2440548, 17.9642544
1: -5.1535125, 13.3938541, -6.8216715, 17.0121117, -22.1656208, 20.2155247
2: -4.2921443, 15.0030842, -5.7241507, 19.0055046, -23.2976494, 20.7272339
3: -5.2045627, 19.3389301, -6.9031906, 24.3999500, -29.6045132, 26.2421207
4: -4.2549839, 17.7536049, -5.5873413, 22.5300140, -26.7849979, 23.3409443

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5246138, upper bound: 27.5204369
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5336368, upper bound: 27.5318549
time: 0.77 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 4.30 seconds
IS_A2_B1_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.4408190, upper bound: 27.4518655
IS_A2_B1_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.4337297, upper bound: 27.4489984
IS_A2_B1_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.4949767, upper bound: 27.4984165
IS_A2_B1_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.4756057, upper bound: 27.4805101
IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.4857993, upper bound: 27.4911214
IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.5275780, upper bound: 27.5294906
IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.4828474, upper bound: 27.4816232
IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.5313305, upper bound: 27.5295223
IS_A2_B2_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.5294437, upper bound: 27.5161280
IS_A2_B2_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.5291661, upper bound: 27.5201675
IS_A2_B2_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.5052457, upper bound: 27.4829690
IS_A2_B2_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.5304591, upper bound: 27.5189092
IS_A2_B2_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.5245080, upper bound: 27.5068773
IS_A2_B2_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.5245080, upper bound: 27.5115666
IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.4994839, upper bound: 27.4788255
IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.5222219, upper bound: 27.5092144
IS_A2_B2_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.5237602, upper bound: 27.5235139
IS_A2_B2_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.5237602, upper bound: 27.5285237
IS_A2_B2_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.4987126, upper bound: 27.4851986
IS_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.4884721, upper bound: 27.4807831
IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.5125649, upper bound: 27.5165320
IS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.5240240, upper bound: 27.5280145
IS_A2_B2_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.5246138, upper bound: 27.5204369
IS_A2_B2_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.30
Output dim: 3, lower bound: -27.5336368, upper bound: 27.5318549

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.1501436, 14.8518286, -4.0484672, 14.4596176, -18.6097603, 18.9002953
1: -5.9463558, 15.1576147, -5.8160443, 14.7340803, -20.6804352, 20.9736595
2: -4.9646659, 16.9373550, -4.8774180, 16.4880371, -21.4527016, 21.8147736
3: -6.0380936, 21.8218441, -5.8613138, 21.1543846, -27.1924725, 27.6831570
4: -4.9109526, 20.0992146, -4.7867885, 19.4765301, -24.3874798, 24.8860016

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 45

Time for candidate selection: 3.20 seconds

### Candidate
type: A, layer: 3, pos: 46

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5250105, upper bound: 27.5238495
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5140203, upper bound: 27.5196626
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.1731067, 14.9586849, -4.0448785, 14.4511614, -18.6242676, 19.0035629
1: -5.9808068, 15.2638369, -5.8107276, 14.7251501, -20.7059555, 21.0745640
2: -4.9876971, 17.0616207, -4.8726258, 16.4781666, -21.4658623, 21.9342461
3: -6.0771651, 21.9848022, -5.8559823, 21.1425705, -27.2197361, 27.8407841
4: -4.9351068, 20.2439060, -4.7826281, 19.4649715, -24.4000778, 25.0265350

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 45

Time for candidate selection: 3.03 seconds

### Candidate
type: A, layer: 3, pos: 46

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5291543, upper bound: 27.5238499
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4824188, upper bound: 27.4853226
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -3.5780735, 13.1506062, -3.7988207, 13.7142839, -17.2923508, 16.9494267
1: -5.1448393, 13.3754005, -5.4307499, 14.0117922, -19.1566315, 18.8061504
2: -4.2847376, 14.9827337, -4.5483894, 15.7165747, -20.0013123, 19.5311241
3: -5.1957798, 19.3137321, -5.5336990, 20.2515450, -25.4473248, 24.8474312
4: -4.2482719, 17.7293530, -4.5293179, 18.6553898, -22.9036617, 22.2586708

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4485609, upper bound: 27.4372521
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4498935, upper bound: 27.4351848
time: 2.30 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -3.5752065, 13.1439962, -3.8228188, 13.8380985, -17.4133015, 16.9668140
1: -5.1405926, 13.3684082, -5.4709215, 14.1353645, -19.2759552, 18.8393288
2: -4.2808051, 14.9750490, -4.5737586, 15.8609829, -20.1417885, 19.5488071
3: -5.1915131, 19.3046131, -5.5750418, 20.4386253, -25.6301327, 24.8796539
4: -4.2448320, 17.7204685, -4.5566120, 18.8219814, -23.0668125, 22.2770786

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4772867, upper bound: 27.4719717
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4778397, upper bound: 27.4682964
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -3.5840602, 13.1680984, -4.4443007, 15.5364027, -19.1204624, 17.6124001
1: -5.1535125, 13.3938541, -6.3097868, 15.8822498, -21.0357590, 19.7036400
2: -4.2921443, 15.0030842, -5.3230910, 17.8039646, -22.0961094, 20.3261757
3: -5.2045627, 19.3389301, -6.3947978, 22.8526421, -28.0572052, 25.7337284
4: -4.2549839, 17.7536049, -5.2203445, 21.1094589, -25.3644428, 22.9739494

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5265194, upper bound: 27.5188081
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5297461, upper bound: 27.5188081
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.0484672, 14.4596176, -3.7988207, 13.7142839, -17.7627487, 18.2584362
1: -5.8160443, 14.7340803, -5.4307499, 14.0117922, -19.8278370, 20.1648293
2: -4.8774180, 16.4880371, -4.5483894, 15.7165747, -20.5939922, 21.0364246
3: -5.8613138, 21.1543846, -5.5336990, 20.2515450, -26.1128578, 26.6880817
4: -4.7867885, 19.4765301, -4.5293179, 18.6553898, -23.4421787, 24.0058479

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5082787, upper bound: 27.4904719
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5221756, upper bound: 27.5046677
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.0448785, 14.4511614, -3.8228188, 13.8380985, -17.8829746, 18.2739792
1: -5.8107276, 14.7251501, -5.4709215, 14.1353645, -19.9460907, 20.1960716
2: -4.8726258, 16.4781666, -4.5737586, 15.8609829, -20.7336082, 21.0519238
3: -5.8559823, 21.1425705, -5.5750418, 20.4386253, -26.2946072, 26.7176132
4: -4.7826281, 19.4649715, -4.5566120, 18.8219814, -23.6046085, 24.0215836

Time for backsubstitution: 2.61 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=31.572500228881836
rel_dist={3: [-27.542158575057197, 27.5421585750572]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1121.68 seconds
