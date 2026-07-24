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
execution time: IAR + LP analysis = 2.47 + 1.96 = 4.44 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -27.5477966, upper bound: 27.5477966


# Binary Search by BASE starts (time budget: 1195.56 seconds, max iter: 100)

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
Binary search time: 71.85 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1123.72 seconds

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5466365, upper bound: 27.5440437
time: 0.76 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5476940, upper bound: 27.5476940
time: 0.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.74 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.74
Output dim: 3, lower bound: -27.5466365, upper bound: 27.5440437
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.74
Output dim: 3, lower bound: -27.5476940, upper bound: 27.5476940

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.1535349, 14.4982834, -4.8926516, 16.6545238, -20.8080578, 19.3909321
1: -6.0371099, 14.8871717, -6.9941020, 17.1100998, -23.1472054, 21.8812714
2: -5.1117692, 16.7495937, -5.9263554, 19.1902447, -24.3020134, 22.6759491
3: -6.0486369, 21.3488541, -7.0618343, 24.5106659, -30.5593033, 28.4106884
4: -4.9890728, 19.7949219, -5.7611084, 22.7441978, -27.7332706, 25.5560303

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5429862, upper bound: 27.5429862
time: 0.68 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5429862, upper bound: 27.5440437
time: 0.57 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -4.8926516, 16.6545238, -21.4577904, 21.2698650
1: -6.8679352, 16.8314762, -6.9941020, 17.1100998, -23.9780331, 23.8255749
2: -5.8185549, 18.8867245, -5.9263554, 19.1902447, -25.0087986, 24.8130779
3: -6.9354429, 24.1179543, -7.0618343, 24.5106659, -31.4461098, 31.1797886
4: -5.6641207, 22.3921146, -5.7611084, 22.7441978, -28.4083176, 28.1532230

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5440437, upper bound: 27.5466365
time: 0.70 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5440437, upper bound: 27.5476940
time: 0.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.85 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.85
Output dim: 3, lower bound: -27.5429862, upper bound: 27.5429862
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.85
Output dim: 3, lower bound: -27.5429862, upper bound: 27.5440437
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.85
Output dim: 3, lower bound: -27.5440437, upper bound: 27.5466365
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.85
Output dim: 3, lower bound: -27.5440437, upper bound: 27.5476940

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -4.1535349, 14.4982834, -4.1535349, 14.4982834, -18.6518173, 18.6518173
1: -6.0371099, 14.8871717, -6.0371099, 14.8871717, -20.9242802, 20.9242802
2: -5.1117692, 16.7495937, -5.1117692, 16.7495937, -21.8613625, 21.8613625
3: -6.0486369, 21.3488541, -6.0486369, 21.3488541, -27.3974915, 27.3974915
4: -4.9890728, 19.7949219, -4.9890728, 19.7949219, -24.7839947, 24.7839947

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5342809, upper bound: 27.5251542
time: 0.77 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5411654, upper bound: 27.5411653
time: 0.70 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -4.1535349, 14.4982834, -4.8032727, 16.3772182, -20.5307503, 19.3015537
1: -6.0371099, 14.8871717, -6.8679352, 16.8314762, -22.8685818, 21.7551079
2: -5.1117692, 16.7495937, -5.8185549, 18.8867245, -23.9984932, 22.5681496
3: -6.0486369, 21.3488541, -6.9354429, 24.1179543, -30.1665916, 28.2842979
4: -4.9890728, 19.7949219, -5.6641207, 22.3921146, -27.3811874, 25.4590416

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5323184, upper bound: 27.5392574
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5323184, upper bound: 27.5424735
time: 0.81 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -4.1535349, 14.4982834, -19.3015556, 20.5307503
1: -6.8679352, 16.8314762, -6.0371099, 14.8871717, -21.7551079, 22.8685818
2: -5.8185549, 18.8867245, -5.1117692, 16.7495937, -22.5681496, 23.9984932
3: -6.9354429, 24.1179543, -6.0486369, 21.3488541, -28.2842979, 30.1665916
4: -5.6641207, 22.3921146, -4.9890728, 19.7949219, -25.4590416, 27.3811874

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5392574, upper bound: 27.5359171
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5392574, upper bound: 27.5452691
time: 0.85 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -4.8032727, 16.3772182, -21.1804867, 21.1804867
1: -6.8679352, 16.8314762, -6.8679352, 16.8314762, -23.6994114, 23.6994114
2: -5.8185549, 18.8867245, -5.8185549, 18.8867245, -24.7052765, 24.7052746
3: -6.9354429, 24.1179543, -6.9354429, 24.1179543, -31.0533962, 31.0533962
4: -5.6641207, 22.3921146, -5.6641207, 22.3921146, -28.0562363, 28.0562363

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5351261, upper bound: 27.5288045
time: 0.60 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5420105, upper bound: 27.5456608
time: 0.79 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.95 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.95
Output dim: 3, lower bound: -27.5342809, upper bound: 27.5251542
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.95
Output dim: 3, lower bound: -27.5411654, upper bound: 27.5411653
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.95
Output dim: 3, lower bound: -27.5323184, upper bound: 27.5392574
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.95
Output dim: 3, lower bound: -27.5323184, upper bound: 27.5424735
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.95
Output dim: 3, lower bound: -27.5392574, upper bound: 27.5359171
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.95
Output dim: 3, lower bound: -27.5392574, upper bound: 27.5452691
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.95
Output dim: 3, lower bound: -27.5351261, upper bound: 27.5288045
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.95
Output dim: 3, lower bound: -27.5420105, upper bound: 27.5456608

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -4.1535349, 14.4982834, -3.8096104, 13.5872917, -17.7408257, 18.3078938
1: -6.0371099, 14.8871717, -5.5293055, 13.9026499, -19.9397583, 20.4164772
2: -5.1117692, 16.7495937, -4.6612096, 15.6515274, -20.7632961, 21.4108028
3: -6.0486369, 21.3488541, -5.5544548, 20.0211716, -26.0698090, 26.9033089
4: -4.9890728, 19.7949219, -4.5881572, 18.5381012, -23.5271740, 24.3830795

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5064620, upper bound: 27.4995374
time: 0.73 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5178871, upper bound: 27.5045526
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -4.1535349, 14.4982834, -4.0815225, 14.3603363, -18.5138702, 18.5798035
1: -6.0371099, 14.8871717, -5.9201736, 14.7163811, -20.7534847, 20.8073463
2: -5.1117692, 16.7495937, -5.0029421, 16.5449162, -21.6566849, 21.7525368
3: -6.0486369, 21.3488541, -5.9411888, 21.1416397, -27.1902771, 27.2900410
4: -4.9890728, 19.7949219, -4.8949690, 19.5565891, -24.5456600, 24.6898918

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5358685, upper bound: 27.5345145
time: 0.74 seconds

## Relational analysis of IS_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5406214, upper bound: 27.5406212
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.8966453, 13.8043938, -4.8032727, 16.3772182, -20.2738609, 18.6076660
1: -5.6560292, 14.1464357, -6.8679352, 16.8314762, -22.4875031, 21.0143700
2: -4.7809715, 15.9183598, -5.8185549, 18.8867245, -23.6676960, 21.7369137
3: -5.6777368, 20.3282623, -6.9354429, 24.1179543, -29.7956905, 27.2637062
4: -4.6884899, 18.8143234, -5.6641207, 22.3921146, -27.0806046, 24.4784431

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5144679, upper bound: 27.5280441
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5144679, upper bound: 27.5353765
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.2670722, 14.9929380, -4.8032727, 16.3772182, -20.6442909, 19.7962093
1: -6.1816254, 15.3483210, -6.8679352, 16.8314762, -23.0131016, 22.2162533
2: -5.2292953, 17.1967506, -5.8185549, 18.8867245, -24.1160202, 23.0153027
3: -6.2076254, 21.9800053, -6.9354429, 24.1179543, -30.3255806, 28.9154472
4: -5.0987215, 20.3454666, -5.6641207, 22.3921146, -27.4908371, 26.0095863

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5384545, upper bound: 27.5330661
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5384545, upper bound: 27.5424732
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -3.8966453, 13.8043938, -18.6076660, 20.2738609
1: -6.8679352, 16.8314762, -5.6560292, 14.1464357, -21.0143700, 22.4875031
2: -5.8185549, 18.8867245, -4.7809715, 15.9183598, -21.7369137, 23.6676960
3: -6.9354429, 24.1179543, -5.6777368, 20.3282623, -27.2637062, 29.7956905
4: -5.6641207, 22.3921146, -4.6884899, 18.8143234, -24.4784431, 27.0806046

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5286491, upper bound: 27.5180666
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5286491, upper bound: 27.5319601
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -4.2670722, 14.9929380, -19.7962093, 20.6442909
1: -6.8679352, 16.8314762, -6.1816254, 15.3483210, -22.2162514, 23.0131016
2: -5.8185549, 18.8867245, -5.2292953, 17.1967506, -23.0153046, 24.1160202
3: -6.9354429, 24.1179543, -6.2076254, 21.9800053, -28.9154472, 30.3255806
4: -5.6641207, 22.3921146, -5.0987215, 20.3454666, -26.0095863, 27.4908371

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5330661, upper bound: 27.5440824
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5330661, upper bound: 27.5452694
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -4.3976188, 15.3468742, -20.1501408, 20.7748356
1: -6.8679352, 16.8314762, -6.2940626, 15.7216339, -22.5895691, 23.1255360
2: -5.8185549, 18.8867245, -5.3095002, 17.6520042, -23.4705563, 24.1962242
3: -6.9354429, 24.1179543, -6.3730483, 22.6263657, -29.5618095, 30.4910030
4: -5.6641207, 22.3921146, -5.2157865, 20.9694633, -26.6335831, 27.6079006

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5097964, upper bound: 27.5067001
time: 0.79 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5209620, upper bound: 27.5083565
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -4.7206116, 16.2323303, -21.0355949, 21.0978241
1: -6.8679352, 16.8314762, -6.7455416, 16.6543579, -23.5222931, 23.5770187
2: -5.8185549, 18.8867245, -5.7037611, 18.6685505, -24.4871063, 24.5904789
3: -6.9354429, 24.1179543, -6.8226271, 23.8968391, -30.8322830, 30.9405785
4: -5.6641207, 22.3921146, -5.5644245, 22.1408634, -27.8049850, 27.9565372

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5366295, upper bound: 27.5379964
time: 0.67 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5438371, upper bound: 27.5452613
time: 0.94 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.16 seconds
IS_A1_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 4.16
Output dim: 3, lower bound: -27.5064620, upper bound: 27.4995374
IS_A1_B1_B1_A2, status: Status.VERIFIED, split count: 4, time: 4.16
Output dim: 3, lower bound: -27.5178871, upper bound: 27.5045526
IS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.16
Output dim: 3, lower bound: -27.5358685, upper bound: 27.5345145
IS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.16
Output dim: 3, lower bound: -27.5406214, upper bound: 27.5406212
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.16
Output dim: 3, lower bound: -27.5144679, upper bound: 27.5280441
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.16
Output dim: 3, lower bound: -27.5144679, upper bound: 27.5353765
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.16
Output dim: 3, lower bound: -27.5384545, upper bound: 27.5330661
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.16
Output dim: 3, lower bound: -27.5384545, upper bound: 27.5424732
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.16
Output dim: 3, lower bound: -27.5286491, upper bound: 27.5180666
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.16
Output dim: 3, lower bound: -27.5286491, upper bound: 27.5319601
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.16
Output dim: 3, lower bound: -27.5330661, upper bound: 27.5440824
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.16
Output dim: 3, lower bound: -27.5330661, upper bound: 27.5452694
IS_A2_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 4.16
Output dim: 3, lower bound: -27.5097964, upper bound: 27.5067001
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.16
Output dim: 3, lower bound: -27.5209620, upper bound: 27.5083565
IS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.16
Output dim: 3, lower bound: -27.5366295, upper bound: 27.5379964
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.16
Output dim: 3, lower bound: -27.5438371, upper bound: 27.5452613

## BFS IS instance: IS_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -4.0553946, 14.2228661, -2.5642190, 9.9206409, -13.9760361, 16.7870846
1: -5.8964753, 14.5981779, -3.7614348, 10.0586987, -15.9551735, 18.3596096
2: -4.9947186, 16.4274673, -3.1901212, 11.3052711, -16.2999897, 19.6175880
3: -5.9086456, 20.9470367, -3.7287681, 14.6306629, -20.5393085, 24.6758041
4: -4.8828316, 19.4205570, -3.2276156, 13.4916935, -18.3745251, 22.6481724

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5223319, upper bound: 27.5275472
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5335374, upper bound: 27.5329883
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -4.1535349, 14.4982834, -4.0725145, 14.3336220, -18.4871540, 18.5707951
1: -6.0371099, 14.8871717, -5.9068017, 14.6883450, -20.7254543, 20.7939739
2: -5.1117692, 16.7495937, -4.9915209, 16.5139713, -21.6257401, 21.7411156
3: -6.0486369, 21.3488541, -5.9278607, 21.1031227, -27.1517601, 27.2767143
4: -4.9890728, 19.7949219, -4.8845754, 19.5199795, -24.5090523, 24.6794968

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5279455, upper bound: 27.5337099
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5391510, upper bound: 27.5391511
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -3.5618644, 12.9171572, -4.8032727, 16.3772182, -19.9390793, 17.7204285
1: -5.1606417, 13.1869688, -6.8679352, 16.8314762, -21.9921131, 20.0549049
2: -4.3412910, 14.8576298, -5.8185549, 18.8867245, -23.2280159, 20.6761856
3: -5.1961222, 19.0373878, -6.9354429, 24.1179543, -29.3140736, 25.9728317
4: -4.2972727, 17.5959225, -5.6641207, 22.3921146, -26.6893883, 23.2600441

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4937742, upper bound: 27.5144937
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4974493, upper bound: 27.5271937
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -3.8377326, 13.7042408, -4.8032727, 16.3772182, -20.2149487, 18.5075111
1: -5.5586572, 14.0156584, -6.8679352, 16.8314762, -22.3901310, 20.8835945
2: -4.6897707, 15.7574348, -5.8185549, 18.8867245, -23.5764923, 21.5759888
3: -5.5885458, 20.1776047, -6.9354429, 24.1179543, -29.7064991, 27.1130486
4: -4.6097212, 18.6277122, -5.6641207, 22.3921146, -27.0018349, 24.2918320

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4949222, upper bound: 27.5154187
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5316205, upper bound: 27.5344258
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.2670722, 14.9929380, -4.5044422, 15.5995150, -19.8665867, 19.4973755
1: -6.1816254, 15.3483210, -6.4420037, 16.0025711, -22.1841965, 21.7903156
2: -5.2292953, 17.1967506, -5.4514475, 17.9574566, -23.1867523, 22.6481972
3: -6.2076254, 21.9800053, -6.5190701, 22.9807549, -29.1883812, 28.4990749
4: -5.0987215, 20.3454666, -5.3348827, 21.2997704, -26.3984909, 25.6803474

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5390535, upper bound: 27.5283612
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5434657, upper bound: 27.5325372
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.2670722, 14.9929380, -4.9108210, 16.8157921, -21.0828648, 19.9037590
1: -6.1816254, 15.3483210, -7.0152240, 17.2347851, -23.4164085, 22.3635387
2: -5.2292953, 17.1967506, -5.9189138, 19.2696590, -24.4989548, 23.1156635
3: -6.2076254, 21.9800053, -7.0767059, 24.6664829, -30.8741074, 29.0567055
4: -5.0987215, 20.3454666, -5.7561159, 22.8612328, -27.9599533, 26.1015816

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5425182, upper bound: 27.5409963
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5421844, upper bound: 27.5364737
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -3.5618644, 12.9171572, -17.7204304, 19.9390793
1: -6.8679352, 16.8314762, -5.1606417, 13.1869688, -20.0549030, 21.9921131
2: -5.8185549, 18.8867245, -4.3412910, 14.8576298, -20.6761856, 23.2280159
3: -6.9354429, 24.1179543, -5.1961222, 19.0373878, -25.9728317, 29.3140736
4: -5.6641207, 22.3921146, -4.2972727, 17.5959225, -23.2600441, 26.6893883

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5144937, upper bound: 27.4974493
time: 0.84 seconds

## Relational analysis of IS_A2_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5271937, upper bound: 27.5168689
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -3.8377326, 13.7042408, -18.5075111, 20.2149487
1: -6.8679352, 16.8314762, -5.5586572, 14.0156584, -20.8835945, 22.3901310
2: -5.8185549, 18.8867245, -4.6897707, 15.7574348, -21.5759888, 23.5764942
3: -6.9354429, 24.1179543, -5.5885458, 20.1776047, -27.1130486, 29.7064991
4: -5.6641207, 22.3921146, -4.6097212, 18.6277122, -24.2918320, 27.0018349

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5154187, upper bound: 27.4949222
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2

### Relational analysis result of IS_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344258, upper bound: 27.5316205
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -4.5044422, 15.5995150, -4.2670722, 14.9929380, -19.4973774, 19.8665867
1: -6.4420037, 16.0025711, -6.1816254, 15.3483210, -21.7903175, 22.1841946
2: -5.4514475, 17.9574566, -5.2292953, 17.1967506, -22.6481972, 23.1867523
3: -6.5190701, 22.9807549, -6.2076254, 21.9800053, -28.4990749, 29.1883812
4: -5.3348827, 21.2997704, -5.0987215, 20.3454666, -25.6803474, 26.3984909

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5132690, upper bound: 27.5390535
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5132690, upper bound: 27.5434657
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -4.9108210, 16.8157921, -4.2670722, 14.9929380, -19.9037571, 21.0828648
1: -7.0152240, 17.2347851, -6.1816254, 15.3483210, -22.3635406, 23.4164104
2: -5.9189138, 19.2696590, -5.2292953, 17.1967506, -23.1156654, 24.4989548
3: -7.0767059, 24.6664829, -6.2076254, 21.9800053, -29.0567055, 30.8741074
4: -5.7561159, 22.8612328, -5.0987215, 20.3454666, -26.1015816, 27.9599533

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5289012, upper bound: 27.5364825
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5243714, upper bound: 27.5386738
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -4.9108210, 16.8157921, -4.3976188, 15.3468742, -20.2576923, 21.2134113
1: -7.0152240, 17.2347851, -6.2940626, 15.7216339, -22.7368565, 23.5288429
2: -5.9189138, 19.2696590, -5.3095002, 17.6520042, -23.5709171, 24.5791588
3: -7.0767059, 24.6664829, -6.3730483, 22.6263657, -29.7030678, 31.0395317
4: -5.7561159, 22.8612328, -5.2157865, 20.9694633, -26.7255783, 28.0770187

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5209623, upper bound: 27.5083565
time: 0.74 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5209623, upper bound: 27.5083565
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -4.6995625, 16.0844040, -3.1353536, 11.4849281, -16.1844883, 19.2197552
1: -6.7196193, 16.5238991, -4.4685645, 11.6871462, -18.4067650, 20.9924603
2: -5.6939111, 18.5423527, -3.7695451, 13.1007948, -18.7947044, 22.3118973
3: -6.7847633, 23.6901455, -4.4673243, 16.9206219, -23.7053852, 28.1574707
4: -5.5508480, 21.9918098, -3.7639630, 15.6148758, -21.1657238, 25.7557735

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5229468, upper bound: 27.5347204
time: 0.84 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5340058, upper bound: 27.5364647
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -4.7071800, 16.1992378, -21.0025063, 21.0843964
1: -6.8679352, 16.8314762, -6.7290516, 16.6196651, -23.4876003, 23.5605278
2: -5.8185549, 18.8867245, -5.6897607, 18.6302376, -24.4487896, 24.5764847
3: -6.9354429, 24.1179543, -6.8060765, 23.8490314, -30.7844734, 30.9240303
4: -5.6641207, 22.3921146, -5.5516272, 22.0954399, -27.7595596, 27.9437408

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5220845, upper bound: 27.5337099
time: 0.87 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5421454, upper bound: 27.5435420
time: 0.83 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.35 seconds
IS_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5223319, upper bound: 27.5275472
IS_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5335374, upper bound: 27.5329883
IS_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5279455, upper bound: 27.5337099
IS_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5391510, upper bound: 27.5391511
IS_A1_B2_A1_A1_A1, status: Status.VERIFIED, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.4937742, upper bound: 27.5144937
IS_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.4974493, upper bound: 27.5271937
IS_A1_B2_A1_A2_A1, status: Status.VERIFIED, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.4949222, upper bound: 27.5154187
IS_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5316205, upper bound: 27.5344258
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5390535, upper bound: 27.5283612
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5434657, upper bound: 27.5325372
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5425182, upper bound: 27.5409963
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5421844, upper bound: 27.5364737
IS_A2_B1_B1_B1_B1, status: Status.VERIFIED, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5144937, upper bound: 27.4974493
IS_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5271937, upper bound: 27.5168689
IS_A2_B1_B1_B2_B1, status: Status.VERIFIED, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5154187, upper bound: 27.4949222
IS_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5344258, upper bound: 27.5316205
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5132690, upper bound: 27.5390535
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5132690, upper bound: 27.5434657
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5289012, upper bound: 27.5364825
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5243714, upper bound: 27.5386738
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5209623, upper bound: 27.5083565
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5209623, upper bound: 27.5083565
IS_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5229468, upper bound: 27.5347204
IS_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5340058, upper bound: 27.5364647
IS_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5220845, upper bound: 27.5337099
IS_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.35
Output dim: 3, lower bound: -27.5421454, upper bound: 27.5435420

## BFS IS instance: IS_A1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -3.7999940, 13.5306606, -2.5642190, 9.9206409, -13.7206345, 16.0948792
1: -5.5165501, 13.8594303, -3.7614348, 10.0586987, -15.5752478, 17.6208630
2: -4.6648207, 15.5982714, -3.1901212, 11.3052711, -15.9700918, 18.7883911
3: -5.5390534, 19.9286804, -3.7287681, 14.6306629, -20.1697159, 23.6574478
4: -4.5828357, 18.4434834, -3.2276156, 13.4916935, -18.0745296, 21.6710987

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4879288, upper bound: 27.5203150
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4879288, upper bound: 27.5198849
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -4.1755185, 14.7274179, -2.5642190, 9.9206409, -14.0961590, 17.2916355
1: -6.0464978, 15.0691118, -3.7614348, 10.0586987, -16.1051941, 18.8305416
2: -5.1161261, 16.8840675, -3.1901212, 11.3052711, -16.4213982, 20.0741863
3: -6.0716419, 21.5914268, -3.7287681, 14.6306629, -20.7023048, 25.3201942
4: -4.9956708, 19.9824505, -3.2276156, 13.4916935, -18.4873638, 23.2100658

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4764224, upper bound: 27.5092173
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4764224, upper bound: 27.5329883
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -3.8966453, 13.8043938, -4.0725145, 14.3336220, -18.2302647, 17.8769073
1: -5.6560292, 14.1464357, -5.9068017, 14.6883450, -20.3443737, 20.0532360
2: -4.7809715, 15.9183598, -4.9915209, 16.5139713, -21.2949429, 20.9098797
3: -5.6777368, 20.3282623, -5.9278607, 21.1031227, -26.7808590, 26.2561226
4: -4.6884899, 18.8143234, -4.8845754, 19.5199795, -24.2084694, 23.6988983

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5093195, upper bound: 27.5320563
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5224077, upper bound: 27.5292195
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -4.2670722, 14.9929380, -4.0725145, 14.3336220, -18.6006927, 19.0654507
1: -6.1816254, 15.3483210, -5.9068017, 14.6883450, -20.8699703, 21.2551193
2: -5.2292953, 17.1967506, -4.9915209, 16.5139713, -21.7432671, 22.1882687
3: -6.2076254, 21.9800053, -5.9278607, 21.1031227, -27.3107491, 27.9078655
4: -5.0987215, 20.3454666, -4.8845754, 19.5199795, -24.6187019, 25.2300396

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5337100, upper bound: 27.5279454
time: 0.77 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5337100, upper bound: 27.5279454
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -3.5523829, 12.8888369, -4.8032727, 16.3772182, -19.9296017, 17.6921082
1: -5.1465392, 13.1572962, -6.8679352, 16.8314762, -21.9780159, 20.0252304
2: -4.3292475, 14.8248634, -5.8185549, 18.8867245, -23.2159691, 20.6434155
3: -5.1820326, 18.9965401, -6.9354429, 24.1179543, -29.2999840, 25.9319839
4: -4.2862988, 17.5571918, -5.6641207, 22.3921146, -26.6784115, 23.2213116

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5152126, upper bound: 27.5161364
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5152126, upper bound: 27.5161364
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -3.8286188, 13.6773224, -4.8032727, 16.3772182, -20.2058334, 18.4805946
1: -5.5450964, 13.9873419, -6.8679352, 16.8314762, -22.3765717, 20.8552761
2: -4.6782236, 15.7261734, -5.8185549, 18.8867245, -23.5649490, 21.5447273
3: -5.5749130, 20.1388111, -6.9354429, 24.1179543, -29.6928673, 27.0742531
4: -4.5991917, 18.5908585, -5.6641207, 22.3921146, -26.9913044, 24.2549782

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_A2_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5299641, upper bound: 27.5233685
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5299641, upper bound: 27.5233685
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -2.8249590, 10.7035341, -4.4017634, 15.3087053, -18.1336613, 15.1052971
1: -4.1144567, 10.8470554, -6.2926464, 15.6975050, -19.8119621, 17.1397018
2: -3.5004230, 12.1523218, -5.3274221, 17.6157398, -21.1161633, 17.4797440
3: -4.0744195, 15.6850214, -6.3688722, 22.5571671, -26.6315823, 22.0538921
4: -3.4995542, 14.4455595, -5.2222848, 20.9034176, -24.4029713, 19.6678448

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5267382, upper bound: 27.5111459
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5349042, upper bound: 27.5231937
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.2554588, 14.9586115, -4.5044422, 15.5995150, -19.8549728, 19.4630489
1: -6.1641321, 15.3125620, -6.4420037, 16.0025711, -22.1666985, 21.7545643
2: -5.2145309, 17.1568089, -5.4514475, 17.9574566, -23.1719875, 22.6082573
3: -6.1905026, 21.9307594, -6.5190701, 22.9807549, -29.1712513, 28.4498291
4: -5.0852590, 20.2982121, -5.3348827, 21.2997704, -26.3850288, 25.6330929

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5221731, upper bound: 27.4894728
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5076372, upper bound: 27.4826913
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -3.8483875, 13.8090839, -4.9108210, 16.8157921, -20.6641769, 18.7199039
1: -5.6051860, 14.0970688, -7.0152240, 17.2347851, -22.8399715, 21.1122932
2: -4.7509747, 15.8073130, -5.9189138, 19.2696590, -24.0206337, 21.7262268
3: -5.6255231, 20.2453957, -7.0767059, 24.6664829, -30.2920055, 27.3220959
4: -4.6657948, 18.7160625, -5.7561159, 22.8612328, -27.5270271, 24.4721794

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5024668, upper bound: 27.5054417
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5415997, upper bound: 27.5389601
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.2006721, 14.8031902, -4.9108210, 16.8157921, -21.0164604, 19.7140121
1: -6.0869379, 15.1494980, -7.0152240, 17.2347851, -23.3217144, 22.1647167
2: -5.1482048, 16.9752731, -5.9189138, 19.2696590, -24.4178638, 22.8941879
3: -6.1115742, 21.7016144, -7.0767059, 24.6664829, -30.7780571, 28.7783165
4: -5.0243750, 20.0811024, -5.7561159, 22.8612328, -27.8856087, 25.8372192

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5277462, upper bound: 27.5240910
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5091244, upper bound: 27.4864111
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -3.5523829, 12.8888369, -17.6921082, 19.9296017
1: -6.8679352, 16.8314762, -5.1465392, 13.1572962, -20.0252304, 21.9780159
2: -5.8185549, 18.8867245, -4.3292475, 14.8248634, -20.6434174, 23.2159710
3: -6.9354429, 24.1179543, -5.1820326, 18.9965401, -25.9319839, 29.2999840
4: -5.6641207, 22.3921146, -4.2862988, 17.5571918, -23.2213116, 26.6784115

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5034345, upper bound: 27.5152126
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5034345, upper bound: 27.5168689
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -3.8286188, 13.6773224, -18.4805946, 20.2058334
1: -6.8679352, 16.8314762, -5.5450964, 13.9873419, -20.8552761, 22.3765717
2: -5.8185549, 18.8867245, -4.6782236, 15.7261734, -21.5447273, 23.5649471
3: -6.9354429, 24.1179543, -5.5749130, 20.1388111, -27.0742531, 29.6928673
4: -5.6641207, 22.3921146, -4.5991917, 18.5908585, -24.2549782, 26.9913044

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5233685, upper bound: 27.5299641
time: 0.86 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5233685, upper bound: 27.5299641
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.4017634, 15.3087053, -2.8249590, 10.7035341, -15.1052971, 18.1336594
1: -6.2926464, 15.6975050, -4.1144567, 10.8470554, -17.1396980, 19.8119621
2: -5.3274221, 17.6157398, -3.5004230, 12.1523218, -17.4797440, 21.1161633
3: -6.3688722, 22.5571671, -4.0744195, 15.6850214, -22.0538921, 26.6315823
4: -5.2222848, 20.9034176, -3.4995542, 14.4455595, -19.6678448, 24.4029713

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5111459, upper bound: 27.5267382
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5231935, upper bound: 27.5349038
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.5044422, 15.5995150, -4.2554588, 14.9586115, -19.4630508, 19.8549728
1: -6.4420037, 16.0025711, -6.1641321, 15.3125620, -21.7545643, 22.1666985
2: -5.4514475, 17.9574566, -5.2145309, 17.1568089, -22.6082573, 23.1719875
3: -6.5190701, 22.9807549, -6.1905026, 21.9307594, -28.4498291, 29.1712532
4: -5.3348827, 21.2997704, -5.0852590, 20.2982121, -25.6330929, 26.3850288

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4894728, upper bound: 27.5221730
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4826913, upper bound: 27.5076373
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.9108210, 16.8157921, -3.8483875, 13.8090839, -18.7199039, 20.6641769
1: -7.0152240, 17.2347851, -5.6051860, 14.0970688, -21.1122932, 22.8399715
2: -5.9189138, 19.2696590, -4.7509747, 15.8073130, -21.7262268, 24.0206337
3: -7.0767059, 24.6664829, -5.6255231, 20.2453957, -27.3220959, 30.2920055
4: -5.7561159, 22.8612328, -4.6657948, 18.7160625, -24.4721794, 27.5270271

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5161202, upper bound: 27.5037233
time: 0.94 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5392087, upper bound: 27.5331525
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.9108210, 16.8157921, -4.2006721, 14.8031902, -19.7140121, 21.0164604
1: -7.0152240, 17.2347851, -6.0869379, 15.1494980, -22.1647167, 23.3217144
2: -5.9189138, 19.2696590, -5.1482048, 16.9752731, -22.8941879, 24.4178638
3: -7.0767059, 24.6664829, -6.1115742, 21.7016144, -28.7783165, 30.7780571
4: -5.7561159, 22.8612328, -5.0243750, 20.0811024, -25.8372192, 27.8856087

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5225434, upper bound: 27.5222897
time: 0.88 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4826782, upper bound: 27.5011701
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.9108210, 16.8157921, -4.1293182, 14.6288443, -19.5396652, 20.9451103
1: -7.0152240, 17.2347851, -5.9055848, 14.9554243, -21.9706478, 23.1403675
2: -5.9189138, 19.2696590, -4.9716229, 16.8073635, -22.7262764, 24.2412815
3: -7.0767059, 24.6664829, -5.9947290, 21.5765266, -28.6532288, 30.6612129
4: -5.7561159, 22.8612328, -4.9104395, 19.9661102, -25.7222252, 27.7716713

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4826532, upper bound: 27.4853801
time: 0.88 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4853473, upper bound: 27.5083565
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.9108210, 16.8157921, -4.5115433, 15.8061476, -20.7169685, 21.3273354
1: -7.0152240, 17.2347851, -6.4535074, 16.1446762, -23.1598969, 23.6882915
2: -5.9189138, 19.2696590, -5.4166827, 18.0737495, -23.9926643, 24.6863422
3: -7.0767059, 24.6664829, -6.5231509, 23.1954803, -30.2721825, 31.1896343
4: -5.7561159, 22.8612328, -5.3110218, 21.4500999, -27.2062149, 28.1722546

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5034345, upper bound: 27.5016332
time: 0.74 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5184003, upper bound: 27.5055277
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -4.4017634, 15.3087053, -3.1353536, 11.4849281, -15.8866920, 18.4440556
1: -6.2926464, 15.6975050, -4.4685645, 11.6871462, -17.9797935, 20.1660671
2: -5.3274221, 17.6157398, -3.7695451, 13.1007948, -18.4282131, 21.3852844
3: -6.3688722, 22.5571671, -4.4673243, 16.9206219, -23.2894936, 27.0244904
4: -5.2222848, 20.9034176, -3.7639630, 15.6148758, -20.8371601, 24.6673813

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4928195, upper bound: 27.5265401
time: 0.86 seconds

## Relational analysis of IS_A2_B2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4879288, upper bound: 27.5326081
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -4.8159933, 16.5394020, -3.1353536, 11.4849281, -16.3009205, 19.6747551
1: -6.8794155, 16.9427795, -4.4685645, 11.6871462, -18.5665627, 21.4113407
2: -5.8032980, 18.9454288, -3.7695451, 13.1007948, -18.9040890, 22.7149734
3: -6.9355869, 24.2603912, -4.4673243, 16.9206219, -23.8562088, 28.7277145
4: -5.6491137, 22.4793644, -3.7639630, 15.6148758, -21.2639885, 26.2433281

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4778410, upper bound: 27.5114812
time: 1.05 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4778410, upper bound: 27.5364647
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -4.5044422, 15.5995150, -4.7071800, 16.1992378, -20.7036762, 20.3066940
1: -6.4420037, 16.0025711, -6.7290516, 16.6196651, -23.0616684, 22.7316227
2: -5.4514475, 17.9574566, -5.6897607, 18.6302376, -24.0816841, 23.6472168
3: -6.5190701, 22.9807549, -6.8060765, 23.8490314, -30.3681011, 29.7868309
4: -5.3348827, 21.2997704, -5.5516272, 22.0954399, -27.4303226, 26.8513985

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4954513, upper bound: 27.5336175
time: 1.07 seconds

## Relational analysis of IS_A2_B2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4954513, upper bound: 27.4956030
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -4.9108210, 16.8157921, -4.7071800, 16.1992378, -21.1100559, 21.5229702
1: -7.0152240, 17.2347851, -6.7290516, 16.6196651, -23.6348896, 23.9638367
2: -5.9189138, 19.2696590, -5.6897607, 18.6302376, -24.5491524, 24.9594193
3: -7.0767059, 24.6664829, -6.8060765, 23.8490314, -30.9257355, 31.4725590
4: -5.7561159, 22.8612328, -5.5516272, 22.0954399, -27.8515549, 28.4128609

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5160183, upper bound: 27.4949222
time: 0.99 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5160183, upper bound: 27.5435421
time: 1.00 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.73 seconds
IS_A1_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.4879288, upper bound: 27.5203150
IS_A1_B1_B2_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.4879288, upper bound: 27.5198849
IS_A1_B1_B2_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.4764224, upper bound: 27.5092173
IS_A1_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.4764224, upper bound: 27.5329883
IS_A1_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5093195, upper bound: 27.5320563
IS_A1_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5224077, upper bound: 27.5292195
IS_A1_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5337100, upper bound: 27.5279454
IS_A1_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5337100, upper bound: 27.5279454
IS_A1_B2_A1_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5152126, upper bound: 27.5161364
IS_A1_B2_A1_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5152126, upper bound: 27.5161364
IS_A1_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5299641, upper bound: 27.5233685
IS_A1_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5299641, upper bound: 27.5233685
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5267382, upper bound: 27.5111459
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5349042, upper bound: 27.5231937
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5221731, upper bound: 27.4894728
IS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5076372, upper bound: 27.4826913
IS_A1_B2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5024668, upper bound: 27.5054417
IS_A1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5415997, upper bound: 27.5389601
IS_A1_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5277462, upper bound: 27.5240910
IS_A1_B2_A2_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5091244, upper bound: 27.4864111
IS_A2_B1_B1_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5034345, upper bound: 27.5152126
IS_A2_B1_B1_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5034345, upper bound: 27.5168689
IS_A2_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5233685, upper bound: 27.5299641
IS_A2_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5233685, upper bound: 27.5299641
IS_A2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5111459, upper bound: 27.5267382
IS_A2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5231935, upper bound: 27.5349038
IS_A2_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.4894728, upper bound: 27.5221730
IS_A2_B1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.4826913, upper bound: 27.5076373
IS_A2_B1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5161202, upper bound: 27.5037233
IS_A2_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5392087, upper bound: 27.5331525
IS_A2_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5225434, upper bound: 27.5222897
IS_A2_B1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.4826782, upper bound: 27.5011701
IS_A2_B2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.4826532, upper bound: 27.4853801
IS_A2_B2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.4853473, upper bound: 27.5083565
IS_A2_B2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5034345, upper bound: 27.5016332
IS_A2_B2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5184003, upper bound: 27.5055277
IS_A2_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.4928195, upper bound: 27.5265401
IS_A2_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.4879288, upper bound: 27.5326081
IS_A2_B2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.4778410, upper bound: 27.5114812
IS_A2_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.4778410, upper bound: 27.5364647
IS_A2_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.4954513, upper bound: 27.5336175
IS_A2_B2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.4954513, upper bound: 27.4956030
IS_A2_B2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5160183, upper bound: 27.4949222
IS_A2_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.73
Output dim: 3, lower bound: -27.5160183, upper bound: 27.5435421

## BFS IS instance: IS_A1_B1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -3.4677086, 12.6491776, -2.5642190, 9.9206409, -13.3883495, 15.2133961
1: -5.0244374, 12.9056015, -3.7614348, 10.0586987, -15.0831356, 16.6670341
2: -4.2273307, 14.5442104, -3.1901212, 11.3052711, -15.5325985, 17.7343311
3: -5.0589113, 18.6463127, -3.7287681, 14.6306629, -19.6895752, 22.3750782
4: -4.1934676, 17.2327690, -3.2276156, 13.4916935, -17.6851616, 20.4603844

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4637952, upper bound: 27.5162612
time: 0.70 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4874568, upper bound: 27.5193568
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -4.0694900, 14.5315752, -2.5642190, 9.9206409, -13.9901285, 17.0957947
1: -5.8878632, 14.8334103, -3.7614348, 10.0586987, -15.9465618, 18.5948410
2: -4.9551077, 16.6011810, -3.1901212, 11.3052711, -16.2603798, 19.7912998
3: -5.9252195, 21.2937450, -3.7287681, 14.6306629, -20.5558815, 25.0225124
4: -4.8578582, 19.6412334, -3.2276156, 13.4916935, -18.3495522, 22.8688488

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4630720, upper bound: 27.5318962
time: 0.73 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4757947, upper bound: 27.5320301
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -3.4778671, 12.6619568, -4.0725145, 14.3336220, -17.8114872, 16.7344704
1: -5.0943027, 12.9275675, -5.9068017, 14.6883450, -19.7826481, 18.8343697
2: -4.3064899, 14.5695906, -4.9915209, 16.5139713, -20.8204594, 19.5611095
3: -5.1121516, 18.6458187, -5.9278607, 21.1031227, -26.2152748, 24.5736790
4: -4.2603827, 17.2402229, -4.8845754, 19.5199795, -23.7803612, 22.1247978

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4631770, upper bound: 27.4841009
time: 0.83 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4697375, upper bound: 27.5128769
time: 0.76 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_B1

### Relational analysis result of IS_A1_B1_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5093195, upper bound: 27.5291192
time: 0.92 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_B2

### Relational analysis result of IS_A1_B1_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5093195, upper bound: 27.5292195
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -3.8283594, 13.6124163, -4.0725145, 14.3336220, -18.1619778, 17.6849308
1: -5.5591779, 13.9450855, -5.9068017, 14.6883450, -20.2475224, 19.8518867
2: -4.6983938, 15.6951733, -4.9915209, 16.5139713, -21.2123642, 20.6866932
3: -5.5803647, 20.0477257, -5.9278607, 21.1031227, -26.6834869, 25.9755859
4: -4.6137924, 18.5494270, -4.8845754, 19.5199795, -24.1337700, 23.4340019

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_A1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4764786, upper bound: 27.4815983
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4388792, upper bound: 27.4402781
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.2670722, 14.9929380, -3.8286188, 13.6773224, -17.9443951, 18.8215561
1: -6.1816254, 15.3483210, -5.5450964, 13.9873419, -20.1689682, 20.8934135
2: -5.2292953, 17.1967506, -4.6782236, 15.7261734, -20.9554691, 21.8749733
3: -6.2076254, 21.9800053, -5.5749130, 20.1388111, -26.3464355, 27.5549183
4: -5.0987215, 20.3454666, -4.5991917, 18.5908585, -23.6895790, 24.9446564

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5320560, upper bound: 27.5093191
time: 0.86 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5292195, upper bound: 27.5224075
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.2670722, 14.9929380, -4.1501856, 14.7619257, -19.0289974, 19.1431236
1: -6.1816254, 15.3483210, -6.0045290, 15.0758657, -21.2574921, 21.3528423
2: -5.2292953, 17.1967506, -5.0533237, 16.8712635, -22.1005592, 22.2500744
3: -6.2076254, 21.9800053, -6.0436320, 21.6307240, -27.8383484, 28.0236359
4: -5.0987215, 20.3454666, -4.9472756, 19.9544792, -25.0531998, 25.2927380

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4738025, upper bound: 27.5193732
time: 0.74 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4567864, upper bound: 27.4773626
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -3.8286188, 13.6773224, -4.5044422, 15.5995150, -19.4281330, 18.1817646
1: -5.5450964, 13.9873419, -6.4420037, 16.0025711, -21.5476685, 20.4293404
2: -4.6782236, 15.7261734, -5.4514475, 17.9574566, -22.6356812, 21.1776199
3: -5.5749130, 20.1388111, -6.5190701, 22.9807549, -28.5556679, 26.6578808
4: -4.5991917, 18.5908585, -5.3348827, 21.2997704, -25.8989601, 23.9257412

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5279197, upper bound: 27.5109523
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5284501, upper bound: 27.5196358
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -3.8286188, 13.6773224, -4.9108210, 16.8157921, -20.6444092, 18.5881424
1: -5.5450964, 13.9873419, -7.0152240, 17.2347851, -22.7798805, 21.0025616
2: -4.6782236, 15.7261734, -5.9189138, 19.2696590, -23.9478836, 21.6450882
3: -5.5749130, 20.1388111, -7.0767059, 24.6664829, -30.2413960, 27.2155113
4: -4.5991917, 18.5908585, -5.7561159, 22.8612328, -27.4604225, 24.3469734

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5144998, upper bound: 27.5216258
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4866870, upper bound: 27.4649397
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -2.8249590, 10.7035341, -4.0357404, 14.3527546, -17.1777096, 14.7392731
1: -4.1144567, 10.8470554, -5.7673779, 14.6666718, -18.7811279, 16.6144333
2: -3.5004230, 12.1523218, -4.8556700, 16.4846039, -19.9850273, 17.0079918
3: -4.0744195, 15.6850214, -5.8558578, 21.1726513, -25.2470646, 21.5408764
4: -3.4995542, 14.4455595, -4.8040190, 19.5905552, -23.0901089, 19.2495785

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5257278, upper bound: 27.4996305
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5257800, upper bound: 27.5103223
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2.8249590, 10.7035341, -4.3354831, 15.2065725, -18.0315323, 15.0390167
1: -4.1144567, 10.8470554, -6.1944551, 15.5656090, -19.6800652, 17.0415096
2: -3.5004230, 12.1523218, -5.2330904, 17.4505177, -20.9509411, 17.3854122
3: -4.0744195, 15.6850214, -6.2777467, 22.4005127, -26.4749279, 21.9627647
4: -3.4995542, 14.4455595, -5.1408305, 20.7113953, -24.2109489, 19.5863895

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5329359, upper bound: 27.5132394
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5339238, upper bound: 27.5223416
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.2554588, 14.9586115, -4.4495568, 15.4599552, -19.7154121, 19.4081650
1: -6.1641321, 15.3125620, -6.3632541, 15.8534613, -22.0175934, 21.6758156
2: -5.2145309, 17.1568089, -5.3820233, 17.7896023, -23.0041332, 22.5388317
3: -6.1905026, 21.9307594, -6.4430094, 22.7762299, -28.9667282, 28.3737679
4: -5.0852590, 20.2982121, -5.2735958, 21.1015110, -26.1867695, 25.5718079

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5215937, upper bound: 27.4825642
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4923074, upper bound: 27.4887661
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -3.7504961, 13.6328163, -4.9108210, 16.8157921, -20.5662861, 18.5436344
1: -5.4568090, 13.8825703, -7.0152240, 17.2347851, -22.6915913, 20.8977909
2: -4.6023803, 15.5473776, -5.9189138, 19.2696590, -23.8720398, 21.4662914
3: -5.4882956, 19.9745960, -7.0767059, 24.6664829, -30.1547756, 27.0512981
4: -4.5374780, 18.4000874, -5.7561159, 22.8612328, -27.3987103, 24.1562042

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4883983, upper bound: 27.5327376
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5408930, upper bound: 27.5384039
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -4.1517515, 14.6799326, -4.9108210, 16.8157921, -20.9675426, 19.5907536
1: -6.0161939, 15.0170507, -7.0152240, 17.2347851, -23.2509785, 22.0322723
2: -5.0848289, 16.8236141, -5.9189138, 19.2696590, -24.3544884, 22.7425270
3: -6.0437131, 21.5192928, -7.0767059, 24.6664829, -30.7101955, 28.5959930
4: -4.9683080, 19.9032288, -5.7561159, 22.8612328, -27.8295403, 25.6593437

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5259189, upper bound: 27.5220387
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5072555, upper bound: 27.4990663
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4661808, upper bound: 27.4603956
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -4.5044422, 15.5995150, -3.8286188, 13.6773224, -18.1817646, 19.4281330
1: -6.4420037, 16.0025711, -5.5450964, 13.9873419, -20.4293423, 21.5476685
2: -5.4514475, 17.9574566, -4.6782236, 15.7261734, -21.1776199, 22.6356792
3: -6.5190701, 22.9807549, -5.5749130, 20.1388111, -26.6578808, 28.5556679
4: -5.3348827, 21.2997704, -4.5991917, 18.5908585, -23.9257412, 25.8989601

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_B2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5109523, upper bound: 27.5279197
time: 0.86 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5196358, upper bound: 27.5284501
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -4.9108210, 16.8157921, -3.8286188, 13.6773224, -18.5881424, 20.6444092
1: -7.0152240, 17.2347851, -5.5450964, 13.9873419, -21.0025616, 22.7798805
2: -5.9189138, 19.2696590, -4.6782236, 15.7261734, -21.6450882, 23.9478836
3: -7.0767059, 24.6664829, -5.5749130, 20.1388111, -27.2155113, 30.2413960
4: -5.7561159, 22.8612328, -4.5991917, 18.5908585, -24.3469734, 27.4604225

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_B2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5171045, upper bound: 27.5159078
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4606717, upper bound: 27.4882972
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.0357404, 14.3527546, -2.8249590, 10.7035341, -14.7392731, 17.1777096
1: -5.7673779, 14.6666718, -4.1144567, 10.8470554, -16.6144333, 18.7811279
2: -4.8556700, 16.4846039, -3.5004230, 12.1523218, -17.0079918, 19.9850273
3: -5.8558578, 21.1726513, -4.0744195, 15.6850214, -21.5408764, 25.2470646
4: -4.8040190, 19.5905552, -3.4995542, 14.4455595, -19.2495785, 23.0901089

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4996305, upper bound: 27.5257278
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5103223, upper bound: 27.5257800
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.3354831, 15.2065725, -2.8249590, 10.7035341, -15.0390167, 18.0315323
1: -6.1944551, 15.5656090, -4.1144567, 10.8470554, -17.0415115, 19.6800652
2: -5.2330904, 17.4505177, -3.5004230, 12.1523218, -17.3854122, 20.9509411
3: -6.2777467, 22.4005127, -4.0744195, 15.6850214, -21.9627647, 26.4749279
4: -5.1408305, 20.7113953, -3.4995542, 14.4455595, -19.5863895, 24.2109489

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5132397, upper bound: 27.5329362
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5223414, upper bound: 27.5339233
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.4495568, 15.4599552, -4.2554588, 14.9586115, -19.4081669, 19.7154102
1: -6.3632541, 15.8534613, -6.1641321, 15.3125620, -21.6758156, 22.0175934
2: -5.3820233, 17.7896023, -5.2145309, 17.1568089, -22.5388317, 23.0041332
3: -6.4430094, 22.7762299, -6.1905026, 21.9307594, -28.3737679, 28.9667282
4: -5.2735958, 21.1015110, -5.0852590, 20.2982121, -25.5718079, 26.1867695

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4825642, upper bound: 27.5215936
time: 0.79 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4887661, upper bound: 27.5216403
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -4.9108210, 16.8157921, -3.7504961, 13.6328163, -18.5436344, 20.5662880
1: -7.0152240, 17.2347851, -5.4568090, 13.8825703, -20.8977909, 22.6915913
2: -5.9189138, 19.2696590, -4.6023803, 15.5473776, -21.4662914, 23.8720398
3: -7.0767059, 24.6664829, -5.4882956, 19.9745960, -27.0512962, 30.1547756
4: -5.7561159, 22.8612328, -4.5374780, 18.4000874, -24.1562042, 27.3987103

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5317857, upper bound: 27.5166737
time: 0.90 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5386653, upper bound: 27.5319471
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -4.9108210, 16.8157921, -4.1517515, 14.6799326, -19.5907536, 20.9675426
1: -7.0152240, 17.2347851, -6.0161939, 15.0170507, -22.0322723, 23.2509785
2: -5.9189138, 19.2696590, -5.0848289, 16.8236141, -22.7425270, 24.3544884
3: -7.0767059, 24.6664829, -6.0437131, 21.5192928, -28.5959930, 30.7101955
4: -5.7561159, 22.8612328, -4.9683080, 19.9032288, -25.6593437, 27.8295403

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5202171, upper bound: 27.5197626
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4965108, upper bound: 27.5030691
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4594100, upper bound: 27.4640054
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -4.0357404, 14.3527546, -3.1353536, 11.4849281, -15.5206680, 17.4881058
1: -5.7673779, 14.6666718, -4.4685645, 11.6871462, -17.4545250, 19.1352367
2: -4.8556700, 16.4846039, -3.7695451, 13.1007948, -17.9564610, 20.2541466
3: -5.8558578, 21.1726513, -4.4673243, 16.9206219, -22.7764797, 25.6399727
4: -4.8040190, 19.5905552, -3.7639630, 15.6148758, -20.4188957, 23.3545170

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4800245, upper bound: 27.5251592
time: 0.92 seconds

## Relational analysis of IS_A2_B2_B2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4907163, upper bound: 27.5252114
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -4.3354831, 15.2065725, -3.1353536, 11.4849281, -15.8204117, 18.3419266
1: -6.1944551, 15.5656090, -4.4685645, 11.6871462, -17.8816013, 20.0341740
2: -5.2330904, 17.4505177, -3.7695451, 13.1007948, -18.3338852, 21.2200623
3: -6.2777467, 22.4005127, -4.4673243, 16.9206219, -23.1983681, 26.8678360
4: -5.1408305, 20.7113953, -3.7639630, 15.6148758, -20.7557068, 24.4753590

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B2_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4637126, upper bound: 27.5293710
time: 0.75 seconds

## Relational analysis of IS_A2_B2_B2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B2_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4907163, upper bound: 27.5312578
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -4.6814065, 16.3125935, -3.1353536, 11.4849281, -16.1663342, 19.4479465
1: -6.6980958, 16.6752625, -4.4685645, 11.6871462, -18.3852425, 21.1438236
2: -5.6272783, 18.6308517, -3.7695451, 13.1007948, -18.7280731, 22.4003963
3: -6.7745366, 23.9193077, -4.4673243, 16.9206219, -23.6951580, 28.3866310
4: -5.5004053, 22.1044388, -3.7639630, 15.6148758, -21.1152802, 25.8684025

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B2_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4697323, upper bound: 27.5351361
time: 0.77 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B2_B2_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4759344, upper bound: 27.4758457
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -4.1293182, 14.6288443, -4.7071800, 16.1992378, -20.3285561, 19.3360233
1: -5.9055848, 14.9554243, -6.7290516, 16.6196651, -22.5252495, 21.6844749
2: -4.9716229, 16.8073635, -5.6897607, 18.6302376, -23.6018562, 22.4971237
3: -5.9947290, 21.5765266, -6.8060765, 23.8490314, -29.8437614, 28.3826027
4: -4.9104395, 19.9661102, -5.5516272, 22.0954399, -27.0058784, 25.5177383

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_B2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_B2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4923015, upper bound: 27.4849975
time: 0.87 seconds

## Relational analysis of IS_A2_B2_B2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_B2_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4923015, upper bound: 27.4849975
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.9108210, 16.8157921, -4.7579708, 16.5425053, -21.4533253, 21.5737591
1: -7.0152240, 17.2347851, -6.8075414, 16.9199429, -23.9351673, 24.0423241
2: -5.9189138, 19.2696590, -5.7232504, 18.9006500, -24.8195648, 24.9929085
3: -7.0767059, 24.6664829, -6.8910208, 24.2590141, -31.3357143, 31.5575027
4: -5.7561159, 22.8612328, -5.5892906, 22.4202957, -28.1764107, 28.4505215

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_B2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5156023, upper bound: 27.5309987
time: 0.90 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5130151, upper bound: 27.5158528
time: 0.92 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.57 seconds
IS_A1_B1_B2_B1_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4637952, upper bound: 27.5162612
IS_A1_B1_B2_B1_A1_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4874568, upper bound: 27.5193568
IS_A1_B1_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4630720, upper bound: 27.5318962
IS_A1_B1_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4757947, upper bound: 27.5320301
IS_A1_B1_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5093195, upper bound: 27.5291192
IS_A1_B1_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5093195, upper bound: 27.5292195
IS_A1_B1_B2_B2_A1_A2_A1, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4764786, upper bound: 27.4815983
IS_A1_B1_B2_B2_A1_A2_A2, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4388792, upper bound: 27.4402781
IS_A1_B1_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5320560, upper bound: 27.5093191
IS_A1_B1_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5292195, upper bound: 27.5224075
IS_A1_B1_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4738025, upper bound: 27.5193732
IS_A1_B1_B2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4567864, upper bound: 27.4773626
IS_A1_B2_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5279197, upper bound: 27.5109523
IS_A1_B2_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5284501, upper bound: 27.5196358
IS_A1_B2_A1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5144998, upper bound: 27.5216258
IS_A1_B2_A1_A2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4866870, upper bound: 27.4649397
IS_A1_B2_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5257278, upper bound: 27.4996305
IS_A1_B2_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5257800, upper bound: 27.5103223
IS_A1_B2_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5329359, upper bound: 27.5132394
IS_A1_B2_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5339238, upper bound: 27.5223416
IS_A1_B2_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5215937, upper bound: 27.4825642
IS_A1_B2_A2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4923074, upper bound: 27.4887661
IS_A1_B2_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4883983, upper bound: 27.5327376
IS_A1_B2_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5408930, upper bound: 27.5384039
IS_A1_B2_A2_B2_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5072555, upper bound: 27.4990663
IS_A1_B2_A2_B2_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4661808, upper bound: 27.4603956
IS_A2_B1_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5109523, upper bound: 27.5279197
IS_A2_B1_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5196358, upper bound: 27.5284501
IS_A2_B1_B1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5171045, upper bound: 27.5159078
IS_A2_B1_B1_B2_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4606717, upper bound: 27.4882972
IS_A2_B1_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4996305, upper bound: 27.5257278
IS_A2_B1_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5103223, upper bound: 27.5257800
IS_A2_B1_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5132397, upper bound: 27.5329362
IS_A2_B1_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5223414, upper bound: 27.5339233
IS_A2_B1_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4825642, upper bound: 27.5215936
IS_A2_B1_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4887661, upper bound: 27.5216403
IS_A2_B1_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5317857, upper bound: 27.5166737
IS_A2_B1_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5386653, upper bound: 27.5319471
IS_A2_B1_B2_A2_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4965108, upper bound: 27.5030691
IS_A2_B1_B2_A2_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4594100, upper bound: 27.4640054
IS_A2_B2_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4800245, upper bound: 27.5251592
IS_A2_B2_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4907163, upper bound: 27.5252114
IS_A2_B2_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4637126, upper bound: 27.5293710
IS_A2_B2_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4907163, upper bound: 27.5312578
IS_A2_B2_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4697323, upper bound: 27.5351361
IS_A2_B2_B2_B1_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4759344, upper bound: 27.4758457
IS_A2_B2_B2_B2_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4923015, upper bound: 27.4849975
IS_A2_B2_B2_B2_A1_A1_B2, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.4923015, upper bound: 27.4849975
IS_A2_B2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5156023, upper bound: 27.5309987
IS_A2_B2_B2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 3, lower bound: -27.5130151, upper bound: 27.5158528

## BFS IS instance: IS_A1_B1_B2_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -4.0389585, 14.4364195, -2.5636170, 9.9188204, -13.9577789, 17.0000362
1: -5.8233781, 14.7200966, -3.7606044, 10.0568056, -15.8801832, 18.4807014
2: -4.8955483, 16.4622803, -3.1893988, 11.3031788, -16.1987267, 19.6516800
3: -5.8635168, 21.1461220, -3.7279236, 14.6280060, -20.4915199, 24.8740463
4: -4.7990799, 19.4879284, -3.2269549, 13.4892349, -18.2883148, 22.7148838

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5147898, upper bound: 27.5282605
time: 0.88 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5150622, upper bound: 27.5272428
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -4.0052023, 14.3459930, -2.5642190, 9.9206409, -13.9258432, 16.9102097
1: -5.7938766, 14.6377563, -3.7614348, 10.0586987, -15.8525753, 18.3991833
2: -4.8749514, 16.3843307, -3.1901212, 11.3052711, -16.1802216, 19.5744476
3: -5.8297024, 21.0244312, -3.7287681, 14.6306629, -20.4603653, 24.7531986
4: -4.7855892, 19.3841286, -3.2276156, 13.4916935, -18.2772827, 22.6117439

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5291951, upper bound: 27.5287898
time: 0.77 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5271296, upper bound: 27.5269977
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -3.4778671, 12.6619568, -3.6548531, 13.1824102, -16.6602726, 16.3168087
1: -5.0943027, 12.9275675, -5.3454638, 13.4625320, -18.5568295, 18.2730312
2: -4.3064899, 14.5695906, -4.5214105, 15.1565170, -19.4630051, 19.0909977
3: -5.1121516, 18.6458187, -5.3615704, 19.4054260, -24.5175762, 24.0073853
4: -4.2603827, 17.2402229, -4.4589601, 17.9307117, -22.1910934, 21.6991825

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4915058, upper bound: 27.5249129
time: 0.76 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5093195, upper bound: 27.5315698
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -3.4778671, 12.6619568, -4.0035143, 14.1380653, -17.6159325, 16.6654701
1: -5.0943027, 12.9275675, -5.8081703, 14.4828968, -19.5771942, 18.7357330
2: -4.3064899, 14.5695906, -4.9071450, 16.2859592, -20.5924435, 19.4767361
3: -5.1121516, 18.6458187, -5.8282070, 20.8169804, -25.9291306, 24.4740257
4: -4.2603827, 17.2402229, -4.8078885, 19.2485218, -23.5089035, 22.0481110

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4915058, upper bound: 27.5250626
time: 1.23 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5093195, upper bound: 27.5317195
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -4.2670722, 14.9929380, -3.4136868, 12.5346861, -16.8017578, 18.4066238
1: -6.1816254, 15.3483210, -4.9875507, 12.7703829, -18.9520073, 20.3358631
2: -5.2292953, 17.1967506, -4.2113910, 14.3788033, -19.6080990, 21.4081421
3: -6.2076254, 21.9800053, -5.0113583, 18.4539299, -24.6615562, 26.9913616
4: -5.0987215, 20.3454666, -4.1760516, 17.0105629, -22.1092815, 24.5215168

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5128767, upper bound: 27.4697370
time: 0.74 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5291191, upper bound: 27.5093187
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5291192, upper bound: 27.5093191
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -4.2670722, 14.9929380, -3.7598753, 13.4838772, -17.7509480, 18.7528114
1: -6.1816254, 15.3483210, -5.4471669, 13.7843618, -19.9659882, 20.7954826
2: -5.2292953, 17.1967506, -4.5944033, 15.5012827, -20.7305775, 21.7911530
3: -6.2076254, 21.9800053, -5.4762363, 19.8563271, -26.0639534, 27.4562416
4: -5.0987215, 20.3454666, -4.5234141, 18.3234520, -23.4221725, 24.8688812

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5101980, upper bound: 27.4834399
time: 0.92 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4567863, upper bound: 27.4574904
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -3.8286188, 13.6773224, -4.0265965, 14.3068142, -18.1354332, 17.7039185
1: -5.5450964, 13.9873419, -5.8039780, 14.6385155, -20.1836128, 19.7913189
2: -4.6782236, 15.7261734, -4.9006457, 16.4482098, -21.1264343, 20.6268196
3: -5.5749130, 20.1388111, -5.8750706, 21.1034126, -26.6783257, 26.0138817
4: -4.5991917, 18.5908585, -4.8428721, 19.5295944, -24.1287842, 23.4337292

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_A2_A2_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5196483, upper bound: 27.4985118
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5196484, upper bound: 27.5108312
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -3.8286188, 13.6773224, -4.3998537, 15.3189535, -19.1475716, 18.0771751
1: -5.5450964, 13.9873419, -6.3017335, 15.7095337, -21.2546310, 20.2890739
2: -4.6782236, 15.7261734, -5.3317151, 17.6310520, -22.3092728, 21.0578880
3: -5.5749130, 20.1388111, -6.3783145, 22.5724220, -28.1473351, 26.5171242
4: -4.5991917, 18.5908585, -5.2270956, 20.9120865, -25.5112762, 23.8179512

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_A2_A2_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5191552, upper bound: 27.5059215
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5191553, upper bound: 27.5190386
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -3.7787068, 13.5479727, -4.9108210, 16.8157921, -20.5944996, 18.4587917
1: -5.4720607, 13.8489351, -7.0152240, 17.2347851, -22.7068424, 20.8641586
2: -4.6131163, 15.5679140, -5.9189138, 19.2696590, -23.8827744, 21.4868259
3: -5.5048809, 19.9485111, -7.0767059, 24.6664829, -30.1713638, 27.0252132
4: -4.5411329, 18.4051971, -5.7561159, 22.8612328, -27.4023666, 24.1613121

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_A2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5008453, upper bound: 27.5210202
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145483, upper bound: 27.5183413
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -2.8243244, 10.7015753, -4.0613732, 14.4318924, -17.2562160, 14.7629461
1: -4.1135683, 10.8450441, -5.7869797, 14.7370214, -18.8505878, 16.6320229
2: -3.4996586, 12.1500721, -4.8742290, 16.5508175, -20.0504742, 17.0243015
3: -4.0735145, 15.6821413, -5.8737588, 21.2684612, -25.3419724, 21.5559006
4: -3.4988475, 14.4428892, -4.8180470, 19.6876526, -23.1864986, 19.2609348

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5131130, upper bound: 27.4780981
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5224586, upper bound: 27.4974525
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -2.8249590, 10.7035341, -3.9547973, 14.1225080, -16.9474678, 14.6583300
1: -4.1144567, 10.8470554, -5.6498070, 14.4240160, -18.5384693, 16.4968624
2: -3.5004230, 12.1523218, -4.7548971, 16.2153568, -19.7157803, 16.9072189
3: -4.0744195, 15.6850214, -5.7379308, 20.8381367, -24.9125519, 21.4229488
4: -3.4995542, 14.4455595, -4.7139463, 19.2721748, -22.7717285, 19.1595058

Time for backsubstitution: 2.58 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=31.572500228881836
rel_dist={3: [-27.547796646245764, 27.54779664624577]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5439552, upper bound: 27.5418003
time: 0.78 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5453620, upper bound: 27.5453619
time: 0.64 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.65 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 3, lower bound: -27.5439552, upper bound: 27.5418003
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 3, lower bound: -27.5453620, upper bound: 27.5453619

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.1535349, 14.4982834, -4.7039499, 16.1093025, -20.2628365, 19.2022324
1: -6.0371099, 14.8871717, -6.7197199, 16.5395355, -22.5766449, 21.6068916
2: -5.1117692, 16.7495937, -5.6901422, 18.5552387, -23.6670074, 22.4397354
3: -6.0486369, 21.3488541, -6.7819815, 23.7121372, -29.7607746, 28.1308346
4: -4.9890728, 19.7949219, -5.5472879, 21.9990082, -26.9880810, 25.3422089

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5405866, upper bound: 27.5405866
time: 0.96 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5405866, upper bound: 27.5418003
time: 0.80 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -4.8858438, 16.6334686, -21.4367409, 21.2630596
1: -6.8679352, 16.8314762, -6.9844136, 17.0888386, -23.9567738, 23.8158894
2: -5.8185549, 18.8867245, -5.9181032, 19.1669827, -24.9855366, 24.8048210
3: -6.9354429, 24.1179543, -7.0520763, 24.4806061, -31.4160500, 31.1700287
4: -5.6641207, 22.3921146, -5.7536016, 22.7172031, -28.3813248, 28.1457157

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5418003, upper bound: 27.5439552
time: 0.62 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5418003, upper bound: 27.5453620
time: 0.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.92 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.92
Output dim: 3, lower bound: -27.5405866, upper bound: 27.5405866
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.92
Output dim: 3, lower bound: -27.5405866, upper bound: 27.5418003
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.92
Output dim: 3, lower bound: -27.5418003, upper bound: 27.5439552
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.92
Output dim: 3, lower bound: -27.5418003, upper bound: 27.5453620

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -4.1535349, 14.4982834, -4.1535349, 14.4982834, -18.6518173, 18.6518173
1: -6.0371099, 14.8871717, -6.0371099, 14.8871717, -20.9242802, 20.9242802
2: -5.1117692, 16.7495937, -5.1117692, 16.7495937, -21.8613625, 21.8613625
3: -6.0486369, 21.3488541, -6.0486369, 21.3488541, -27.3974915, 27.3974915
4: -4.9890728, 19.7949219, -4.9890728, 19.7949219, -24.7839947, 24.7839947

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5226259, upper bound: 27.5323993
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387279, upper bound: 27.5387280
time: 0.81 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -4.1535349, 14.4982834, -4.8032727, 16.3772182, -20.5307503, 19.3015537
1: -6.0371099, 14.8871717, -6.8679352, 16.8314762, -22.8685818, 21.7551079
2: -5.1117692, 16.7495937, -5.8185549, 18.8867245, -23.9984932, 22.5681496
3: -6.0486369, 21.3488541, -6.9354429, 24.1179543, -30.1665916, 28.2842979
4: -4.9890728, 19.7949219, -5.6641207, 22.3921146, -27.3811874, 25.4590416

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5300665, upper bound: 27.5367609
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5300665, upper bound: 27.5401633
time: 0.80 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -4.1535349, 14.4982834, -19.3015556, 20.5307503
1: -6.8679352, 16.8314762, -6.0371099, 14.8871717, -21.7551079, 22.8685818
2: -5.8185549, 18.8867245, -5.1117692, 16.7495937, -22.5681496, 23.9984932
3: -6.9354429, 24.1179543, -6.0486369, 21.3488541, -28.2842979, 30.1665916
4: -5.6641207, 22.3921146, -4.9890728, 19.7949219, -25.4590416, 27.3811874

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5367609, upper bound: 27.5335042
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5367609, upper bound: 27.5425116
time: 0.85 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -4.8032727, 16.3772182, -21.1804867, 21.1804867
1: -6.8679352, 16.8314762, -6.8679352, 16.8314762, -23.6994114, 23.6994114
2: -5.8185549, 18.8867245, -5.8185549, 18.8867245, -24.7052765, 24.7052746
3: -6.9354429, 24.1179543, -6.9354429, 24.1179543, -31.0533962, 31.0533962
4: -5.6641207, 22.3921146, -5.6641207, 22.3921146, -28.0562363, 28.0562363

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5237309, upper bound: 27.5355029
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5397495, upper bound: 27.5433274
time: 0.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.15 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 3, lower bound: -27.5226259, upper bound: 27.5323993
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 3, lower bound: -27.5387279, upper bound: 27.5387280
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 3, lower bound: -27.5300665, upper bound: 27.5367609
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 3, lower bound: -27.5300665, upper bound: 27.5401633
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 3, lower bound: -27.5367609, upper bound: 27.5335042
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 3, lower bound: -27.5367609, upper bound: 27.5425116
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 3, lower bound: -27.5237309, upper bound: 27.5355029
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 3, lower bound: -27.5397495, upper bound: 27.5433274

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.8096104, 13.5872917, -4.1535349, 14.4982834, -18.3078938, 17.7408257
1: -5.5293055, 13.9026499, -6.0371099, 14.8871717, -20.4164772, 19.9397583
2: -4.6612096, 15.6515274, -5.1117692, 16.7495937, -21.4108028, 20.7632942
3: -5.5544548, 20.0211716, -6.0486369, 21.3488541, -26.9033089, 26.0698090
4: -4.5881572, 18.5381012, -4.9890728, 19.7949219, -24.3830776, 23.5271740

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4971426, upper bound: 27.5030332
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5021641, upper bound: 27.5177530
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.0815225, 14.3603363, -4.1535349, 14.4982834, -18.5798054, 18.5138702
1: -5.9201736, 14.7163811, -6.0371099, 14.8871717, -20.8073463, 20.7534828
2: -5.0029421, 16.5449162, -5.1117692, 16.7495937, -21.7525368, 21.6566830
3: -5.9411888, 21.1416397, -6.0486369, 21.3488541, -27.2900410, 27.1902771
4: -4.8949690, 19.5565891, -4.9890728, 19.7949219, -24.6898918, 24.5456600

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5320322, upper bound: 27.5348656
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5382025, upper bound: 27.5382026
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.8966453, 13.8043938, -4.8032727, 16.3772182, -20.2738609, 18.6076660
1: -5.6560292, 14.1464357, -6.8679352, 16.8314762, -22.4875031, 21.0143700
2: -4.7809715, 15.9183598, -5.8185549, 18.8867245, -23.6676960, 21.7369137
3: -5.6777368, 20.3282623, -6.9354429, 24.1179543, -29.7956905, 27.2637062
4: -4.6884899, 18.8143234, -5.6641207, 22.3921146, -27.0806046, 24.4784431

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5152637, upper bound: 27.5269995
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5120794, upper bound: 27.5330525
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.2670722, 14.9929380, -4.7877316, 16.3375168, -20.6045895, 19.7806702
1: -6.1816254, 15.3483210, -6.8463225, 16.7891312, -22.9707565, 22.1946373
2: -5.2292953, 17.1967506, -5.8001766, 18.8387680, -24.0680637, 22.9969273
3: -6.2076254, 21.9800053, -6.9142232, 24.0592709, -30.2668953, 28.8942242
4: -5.0987215, 20.3454666, -5.6475868, 22.3360119, -27.4347343, 25.9930496

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5407282, upper bound: 27.5311131
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5407282, upper bound: 27.5311131
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -3.8966453, 13.8043938, -18.6076660, 20.2738609
1: -6.8679352, 16.8314762, -5.6560292, 14.1464357, -21.0143700, 22.4875031
2: -5.8185549, 18.8867245, -4.7809715, 15.9183598, -21.7369137, 23.6676960
3: -6.9354429, 24.1179543, -5.6777368, 20.3282623, -27.2637062, 29.7956905
4: -5.6641207, 22.3921146, -4.6884899, 18.8143234, -24.4784431, 27.0806046

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5269995, upper bound: 27.5152637
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5330525, upper bound: 27.5292411
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -4.7877316, 16.3375168, -4.2670722, 14.9929380, -19.7806683, 20.6045895
1: -6.8463225, 16.7891312, -6.1816254, 15.3483210, -22.1946373, 22.9707565
2: -5.8001766, 18.8387680, -5.2292953, 17.1967506, -22.9969273, 24.0680637
3: -6.9142232, 24.0592709, -6.2076254, 21.9800053, -28.8942223, 30.2668953
4: -5.6475868, 22.3360119, -5.0987215, 20.3454666, -25.9930496, 27.4347343

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5311131, upper bound: 27.5407282
time: 0.84 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5311131, upper bound: 27.5425119
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.3976188, 15.3468742, -4.8032727, 16.3772182, -20.7748356, 20.1501408
1: -6.2940626, 15.7216339, -6.8679352, 16.8314762, -23.1255360, 22.5895691
2: -5.3095002, 17.6520042, -5.8185549, 18.8867245, -24.1962242, 23.4705544
3: -6.3730483, 22.6263657, -6.9354429, 24.1179543, -30.4910030, 29.5618095
4: -5.2157865, 20.9694633, -5.6641207, 22.3921146, -27.6079006, 26.6335831

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5009087, upper bound: 27.5074518
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5045579, upper bound: 27.5209955
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.7206116, 16.2323303, -4.8032727, 16.3772182, -21.0978241, 21.0355968
1: -6.7455416, 16.6543579, -6.8679352, 16.8314762, -23.5770187, 23.5222931
2: -5.7037611, 18.6685505, -5.8185549, 18.8867245, -24.5904808, 24.4871063
3: -6.8226271, 23.8968391, -6.9354429, 24.1179543, -30.9405785, 30.8322830
4: -5.5644245, 22.1408634, -5.6641207, 22.3921146, -27.9565372, 27.8049850

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5318367, upper bound: 27.5344240
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5417223, upper bound: 27.5428810
time: 0.83 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.19 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.4971426, upper bound: 27.5030332
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5021641, upper bound: 27.5177530
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5320322, upper bound: 27.5348656
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5382025, upper bound: 27.5382026
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5152637, upper bound: 27.5269995
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5120794, upper bound: 27.5330525
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5407282, upper bound: 27.5311131
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5407282, upper bound: 27.5311131
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5269995, upper bound: 27.5152637
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5330525, upper bound: 27.5292411
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5311131, upper bound: 27.5407282
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5311131, upper bound: 27.5425119
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5009087, upper bound: 27.5074518
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5045579, upper bound: 27.5209955
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5318367, upper bound: 27.5344240
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -27.5417223, upper bound: 27.5428810

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -2.5642190, 9.9206409, -3.8758962, 13.7125874, -16.2768059, 13.7965364
1: -3.7614348, 10.0586987, -5.6375475, 14.0625820, -17.8240147, 15.6962442
2: -3.1901212, 11.3052711, -4.7790165, 15.8304939, -19.0206108, 16.0842876
3: -3.7287681, 14.6306629, -5.6504993, 20.2014046, -23.9301720, 20.2811623
4: -3.2276156, 13.4916935, -4.6861649, 18.7288418, -21.9564571, 18.1778584

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5245683, upper bound: 27.5138796
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5306761, upper bound: 27.5327639
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -4.0725145, 14.3336220, -4.1535349, 14.4982834, -18.5707970, 18.4871540
1: -5.9068017, 14.6883450, -6.0371099, 14.8871717, -20.7939739, 20.7254543
2: -4.9915209, 16.5139713, -5.1117692, 16.7495937, -21.7411156, 21.6257401
3: -5.9278607, 21.1031227, -6.0486369, 21.3488541, -27.2767143, 27.1517601
4: -4.8845754, 19.5199795, -4.9890728, 19.7949219, -24.6794968, 24.5090523

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5313139, upper bound: 27.5256874
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5368389, upper bound: 27.5368388
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -3.5618644, 12.9171572, -4.8032727, 16.3772182, -19.9390793, 17.7204285
1: -5.1606417, 13.1869688, -6.8679352, 16.8314762, -21.9921131, 20.0549049
2: -4.3412910, 14.8576298, -5.8185549, 18.8867245, -23.2280159, 20.6761856
3: -5.1961222, 19.0373878, -6.9354429, 24.1179543, -29.3140736, 25.9728317
4: -4.2972727, 17.5959225, -5.6641207, 22.3921146, -26.6893883, 23.2600441

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4914619, upper bound: 27.5134588
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4941393, upper bound: 27.5255420
time: 1.29 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -3.8377326, 13.7042408, -4.8032727, 16.3772182, -20.2149487, 18.5075111
1: -5.5586572, 14.0156584, -6.8679352, 16.8314762, -22.3901310, 20.8835945
2: -4.6897707, 15.7574348, -5.8185549, 18.8867245, -23.5764923, 21.5759888
3: -5.5885458, 20.1776047, -6.9354429, 24.1179543, -29.7064991, 27.1130486
4: -4.6097212, 18.6277122, -5.6641207, 22.3921146, -27.0018349, 24.2918320

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4932852, upper bound: 27.5139957
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5287923, upper bound: 27.5320682
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.2670722, 14.9929380, -4.5044422, 15.5995150, -19.8665867, 19.4973755
1: -6.1816254, 15.3483210, -6.4420037, 16.0025711, -22.1841965, 21.7903156
2: -5.2292953, 17.1967506, -5.4514475, 17.9574566, -23.1867523, 22.6481972
3: -6.2076254, 21.9800053, -6.5190701, 22.9807549, -29.1883812, 28.4990749
4: -5.0987215, 20.3454666, -5.3348827, 21.2997704, -26.3984909, 25.6803474

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5328142, upper bound: 27.5239898
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5402617, upper bound: 27.5305814
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.2670722, 14.9929380, -4.9108210, 16.8157921, -21.0828648, 19.9037590
1: -6.1816254, 15.3483210, -7.0152240, 17.2347851, -23.4164085, 22.3635387
2: -5.2292953, 17.1967506, -5.9189138, 19.2696590, -24.4989548, 23.1156635
3: -6.2076254, 21.9800053, -7.0767059, 24.6664829, -30.8741074, 29.0567055
4: -5.0987215, 20.3454666, -5.7561159, 22.8612328, -27.9599533, 26.1015816

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5010190, upper bound: 27.5036456
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5384948, upper bound: 27.5273031
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -3.5618644, 12.9171572, -17.7204304, 19.9390793
1: -6.8679352, 16.8314762, -5.1606417, 13.1869688, -20.0549030, 21.9921131
2: -5.8185549, 18.8867245, -4.3412910, 14.8576298, -20.6761856, 23.2280159
3: -6.9354429, 24.1179543, -5.1961222, 19.0373878, -25.9728317, 29.3140736
4: -5.6641207, 22.3921146, -4.2972727, 17.5959225, -23.2600441, 26.6893883

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5134588, upper bound: 27.4941393
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5255420, upper bound: 27.5140094
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -3.8377326, 13.7042408, -18.5075111, 20.2149487
1: -6.8679352, 16.8314762, -5.5586572, 14.0156584, -20.8835945, 22.3901310
2: -5.8185549, 18.8867245, -4.6897707, 15.7574348, -21.5759888, 23.5764942
3: -6.9354429, 24.1179543, -5.5885458, 20.1776047, -27.1130486, 29.7064991
4: -5.6641207, 22.3921146, -4.6097212, 18.6277122, -24.2918320, 27.0018349

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5139957, upper bound: 27.4932852
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2

### Relational analysis result of IS_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5320682, upper bound: 27.5287923
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -4.5044422, 15.5995150, -4.2670722, 14.9929380, -19.4973774, 19.8665867
1: -6.4420037, 16.0025711, -6.1816254, 15.3483210, -21.7903175, 22.1841946
2: -5.4514475, 17.9574566, -5.2292953, 17.1967506, -22.6481972, 23.1867523
3: -6.5190701, 22.9807549, -6.2076254, 21.9800053, -28.4990749, 29.1883812
4: -5.3348827, 21.2997704, -5.0987215, 20.3454666, -25.6803474, 26.3984909

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5080322, upper bound: 27.5328139
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5271206, upper bound: 27.5402617
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -4.9108210, 16.8157921, -4.2670722, 14.9929380, -19.9037571, 21.0828648
1: -7.0152240, 17.2347851, -6.1816254, 15.3483210, -22.3635406, 23.4164104
2: -5.9189138, 19.2696590, -5.2292953, 17.1967506, -23.1156654, 24.4989548
3: -7.0767059, 24.6664829, -6.2076254, 21.9800053, -29.0567055, 30.8741074
4: -5.7561159, 22.8612328, -5.0987215, 20.3454666, -26.1015816, 27.9599533

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5036456, upper bound: 27.5052087
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5223561, upper bound: 27.5384217
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.3819494, 15.3058167, -4.9108210, 16.8157921, -21.1977425, 20.2166367
1: -6.2730622, 15.6779966, -7.0152240, 17.2347851, -23.5078430, 22.6932182
2: -5.2906437, 17.6027012, -5.9189138, 19.2696590, -24.5603027, 23.5216141
3: -6.3513303, 22.5658340, -7.0767059, 24.6664829, -31.0178127, 29.6425343
4: -5.1987171, 20.9113693, -5.7561159, 22.8612328, -28.0599499, 26.6674843

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5045580, upper bound: 27.5209957
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5045580, upper bound: 27.5209957
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -3.1353536, 11.4849281, -4.5171452, 15.5555172, -18.6908703, 16.0020733
1: -4.4685645, 11.6871462, -6.4617796, 15.9690351, -20.4375992, 18.1489239
2: -3.7695451, 13.1007948, -5.4726119, 17.9208508, -21.6903954, 18.5734043
3: -4.4673243, 16.9206219, -6.5171919, 22.9176483, -27.3849716, 23.4378109
4: -3.7639630, 15.6148758, -5.3472323, 21.2690334, -25.0329952, 20.9621086

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5244519, upper bound: 27.5136663
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5244519, upper bound: 27.5136663
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -4.7071800, 16.1992378, -4.8032727, 16.3772182, -21.0843964, 21.0025063
1: -6.7290516, 16.6196651, -6.8679352, 16.8314762, -23.5605278, 23.4876003
2: -5.6897607, 18.6302376, -5.8185549, 18.8867245, -24.5764847, 24.4487896
3: -6.8060765, 23.8490314, -6.9354429, 24.1179543, -30.9240303, 30.7844734
4: -5.5516272, 22.0954399, -5.6641207, 22.3921146, -27.9437408, 27.7595596

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5370514, upper bound: 27.5305277
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5396461, upper bound: 27.5409707
time: 0.81 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.33 seconds
IS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5245683, upper bound: 27.5138796
IS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5306761, upper bound: 27.5327639
IS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5313139, upper bound: 27.5256874
IS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5368389, upper bound: 27.5368388
IS_A1_B2_A1_A1_A1, status: Status.VERIFIED, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.4914619, upper bound: 27.5134588
IS_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.4941393, upper bound: 27.5255420
IS_A1_B2_A1_A2_A1, status: Status.VERIFIED, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.4932852, upper bound: 27.5139957
IS_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5287923, upper bound: 27.5320682
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5328142, upper bound: 27.5239898
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5402617, upper bound: 27.5305814
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5010190, upper bound: 27.5036456
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5384948, upper bound: 27.5273031
IS_A2_B1_B1_B1_B1, status: Status.VERIFIED, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5134588, upper bound: 27.4941393
IS_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5255420, upper bound: 27.5140094
IS_A2_B1_B1_B2_B1, status: Status.VERIFIED, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5139957, upper bound: 27.4932852
IS_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5320682, upper bound: 27.5287923
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5080322, upper bound: 27.5328139
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5271206, upper bound: 27.5402617
IS_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5036456, upper bound: 27.5052087
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5223561, upper bound: 27.5384217
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5045580, upper bound: 27.5209957
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5045580, upper bound: 27.5209957
IS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5244519, upper bound: 27.5136663
IS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5244519, upper bound: 27.5136663
IS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5370514, upper bound: 27.5305277
IS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 3, lower bound: -27.5396461, upper bound: 27.5409707

## BFS IS instance: IS_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -2.5642190, 9.9206409, -3.6202762, 13.0224619, -15.5866814, 13.5409174
1: -3.7614348, 10.0586987, -5.2584314, 13.3258266, -17.0872593, 15.3171291
2: -3.1901212, 11.3052711, -4.4492903, 15.0033665, -18.1934853, 15.7545605
3: -3.7287681, 14.6306629, -5.2802815, 19.1860714, -22.9148388, 19.9109440
4: -3.2276156, 13.4916935, -4.3864841, 17.7554073, -20.9830227, 17.8781776

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5186280, upper bound: 27.5034970
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5186283, upper bound: 27.5138797
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -2.5540318, 9.8917933, -4.0042415, 14.2272091, -16.7812405, 13.8960342
1: -3.7471590, 10.0278521, -5.7933350, 14.5435362, -18.2906914, 15.8211870
2: -3.1771150, 11.2710323, -4.9038625, 16.2945766, -19.4716911, 16.1748924
3: -3.7148616, 14.5879736, -5.8166552, 20.8608170, -24.5756760, 20.4046249
4: -3.2156830, 13.4506903, -4.8016357, 19.2998657, -22.5155487, 18.2523251

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5091115, upper bound: 27.4954060
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5091117, upper bound: 27.5300555
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -4.0725145, 14.3336220, -3.8966453, 13.8043938, -17.8769073, 18.2302666
1: -5.9068017, 14.6883450, -5.6560292, 14.1464357, -20.0532360, 20.3443737
2: -4.9915209, 16.5139713, -4.7809715, 15.9183598, -20.9098797, 21.2949429
3: -5.9278607, 21.1031227, -5.6777368, 20.3282623, -26.2561226, 26.7808590
4: -4.8845754, 19.5199795, -4.6884899, 18.8143234, -23.6988983, 24.2084694

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5248261, upper bound: 27.5108817
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5186283, upper bound: 27.5108817
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -4.0574417, 14.2945910, -4.2670722, 14.9929380, -19.0503788, 18.5616627
1: -5.8857427, 14.6467180, -6.1816254, 15.3483210, -21.2340584, 20.8283424
2: -4.9732323, 16.4668427, -5.2292953, 17.1967506, -22.1699829, 21.6961384
3: -5.9075131, 21.0457001, -6.2076254, 21.9800053, -27.8875179, 27.2533264
4: -4.8680658, 19.4648228, -5.0987215, 20.3454666, -25.2135315, 24.5635452

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5256873, upper bound: 27.5313140
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5256873, upper bound: 27.5368391
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -3.5523829, 12.8888369, -4.8032727, 16.3772182, -19.9296017, 17.6921082
1: -5.1465392, 13.1572962, -6.8679352, 16.8314762, -21.9780159, 20.0252304
2: -4.3292475, 14.8248634, -5.8185549, 18.8867245, -23.2159691, 20.6434155
3: -5.1820326, 18.9965401, -6.9354429, 24.1179543, -29.2999840, 25.9319839
4: -4.2862988, 17.5571918, -5.6641207, 22.3921146, -26.6784115, 23.2213116

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5053566, upper bound: 27.5144850
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5053566, upper bound: 27.5136115
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -3.8286188, 13.6773224, -4.8032727, 16.3772182, -20.2058334, 18.4805946
1: -5.5450964, 13.9873419, -6.8679352, 16.8314762, -22.3765717, 20.8552761
2: -4.6782236, 15.7261734, -5.8185549, 18.8867245, -23.5649490, 21.5447273
3: -5.5749130, 20.1388111, -6.9354429, 24.1179543, -29.6928673, 27.0742531
4: -4.5991917, 18.5908585, -5.6641207, 22.3921146, -26.9913044, 24.2549782

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_A2_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5271045, upper bound: 27.5215295
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5271045, upper bound: 27.5215295
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -2.8249590, 10.7035341, -4.2149129, 14.7737617, -17.5987167, 14.9184456
1: -4.1144567, 10.8470554, -6.0240598, 15.1364594, -19.2509155, 16.8711147
2: -3.5004230, 12.1523218, -5.0998349, 16.9875908, -20.4880142, 17.2521553
3: -4.0744195, 15.6850214, -6.0953846, 21.7781258, -25.8525429, 21.7804031
4: -3.4995542, 14.4455595, -5.0150137, 20.1749516, -23.6745052, 19.4605732

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5217158, upper bound: 27.5048244
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5272600, upper bound: 27.5141705
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.2554588, 14.9586115, -4.5044422, 15.5995150, -19.8549728, 19.4630489
1: -6.1641321, 15.3125620, -6.4420037, 16.0025711, -22.1666985, 21.7545643
2: -5.2145309, 17.1568089, -5.4514475, 17.9574566, -23.1719875, 22.6082573
3: -6.1905026, 21.9307594, -6.5190701, 22.9807549, -29.1712513, 28.4498291
4: -5.0852590, 20.2982121, -5.3348827, 21.2997704, -26.3850288, 25.6330929

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5300855, upper bound: 27.5144373
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5380637, upper bound: 27.5268780
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.1616135, 14.7958717, -4.9108210, 16.8157921, -20.9774017, 19.7066917
1: -6.0217419, 15.1112652, -7.0152240, 17.2347851, -23.2565231, 22.1264877
2: -5.0678825, 16.9106216, -5.9189138, 19.2696590, -24.3375416, 22.8295364
3: -6.0606642, 21.6794033, -7.0767059, 24.6664829, -30.7271461, 28.7561035
4: -4.9605665, 20.0012703, -5.7561159, 22.8612328, -27.8218002, 25.7573853

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4871268, upper bound: 27.4949605
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5401627, upper bound: 27.5373908
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -3.5523829, 12.8888369, -17.6921082, 19.9296017
1: -6.8679352, 16.8314762, -5.1465392, 13.1572962, -20.0252304, 21.9780159
2: -5.8185549, 18.8867245, -4.3292475, 14.8248634, -20.6434174, 23.2159710
3: -6.9354429, 24.1179543, -5.1820326, 18.9965401, -25.9319839, 29.2999840
4: -5.6641207, 22.3921146, -4.2862988, 17.5571918, -23.2213116, 26.6784115

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5144850, upper bound: 27.5122461
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5144850, upper bound: 27.5140094
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -3.8286188, 13.6773224, -18.4805946, 20.2058334
1: -6.8679352, 16.8314762, -5.5450964, 13.9873419, -20.8552761, 22.3765717
2: -5.8185549, 18.8867245, -4.6782236, 15.7261734, -21.5447273, 23.5649471
3: -6.9354429, 24.1179543, -5.5749130, 20.1388111, -27.0742531, 29.6928673
4: -5.6641207, 22.3921146, -4.5991917, 18.5908585, -24.2549782, 26.9913044

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4949605, upper bound: 27.5271045
time: 0.98 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4949605, upper bound: 27.5271045
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.2149129, 14.7737617, -2.8249590, 10.7035341, -14.9184456, 17.5987186
1: -6.0240598, 15.1364594, -4.1144567, 10.8470554, -16.8711147, 19.2509155
2: -5.0998349, 16.9875908, -3.5004230, 12.1523218, -17.2521553, 20.4880142
3: -6.0953846, 21.7781258, -4.0744195, 15.6850214, -21.7804031, 25.8525429
4: -5.0150137, 20.1749516, -3.4995542, 14.4455595, -19.4605732, 23.6745033

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4854858, upper bound: 27.4812020
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5141707, upper bound: 27.5272604
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.5044422, 15.5995150, -4.2554588, 14.9586115, -19.4630508, 19.8549728
1: -6.4420037, 16.0025711, -6.1641321, 15.3125620, -21.7545643, 22.1666985
2: -5.4514475, 17.9574566, -5.2145309, 17.1568089, -22.6082573, 23.1719875
3: -6.5190701, 22.9807549, -6.1905026, 21.9307594, -28.4498291, 29.1712532
4: -5.3348827, 21.2997704, -5.0852590, 20.2982121, -25.6330929, 26.3850288

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5144373, upper bound: 27.5300855
time: 0.90 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5268780, upper bound: 27.5380637
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.9108210, 16.8157921, -4.1616135, 14.7958717, -19.7066917, 20.9774017
1: -7.0152240, 17.2347851, -6.0217419, 15.1112652, -22.1264877, 23.2565231
2: -5.9189138, 19.2696590, -5.0678825, 16.9106216, -22.8295364, 24.3375416
3: -7.0767059, 24.6664829, -6.0606642, 21.6794033, -28.7561035, 30.7271461
4: -5.7561159, 22.8612328, -4.9605665, 20.0012703, -25.7573853, 27.8218002

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_B2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4949605, upper bound: 27.4871268
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5375865, upper bound: 27.5380066
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.1293182, 14.6288443, -4.9108210, 16.8157921, -20.9451084, 19.5396652
1: -5.9055848, 14.9554243, -7.0152240, 17.2347851, -23.1403675, 21.9706478
2: -4.9716229, 16.8073635, -5.9189138, 19.2696590, -24.2412815, 22.7262764
3: -5.9947290, 21.5765266, -7.0767059, 24.6664829, -30.6612129, 28.6532307
4: -4.9104395, 19.9661102, -5.7561159, 22.8612328, -27.7716713, 25.7222252

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4853473, upper bound: 27.4853801
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4853473, upper bound: 27.4853801
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.5115433, 15.8061476, -4.9108210, 16.8157921, -21.3273354, 20.7169685
1: -6.4535074, 16.1446762, -7.0152240, 17.2347851, -23.6882915, 23.1598969
2: -5.4166827, 18.0737495, -5.9189138, 19.2696590, -24.6863422, 23.9926643
3: -6.5231509, 23.1954803, -7.0767059, 24.6664829, -31.1896343, 30.2721825
4: -5.3110218, 21.4500999, -5.7561159, 22.8612328, -28.1722546, 27.2062149

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1_B2_A2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4856493, upper bound: 27.4990402
time: 1.40 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4978273, upper bound: 27.5042590
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -3.1353536, 11.4849281, -4.2149129, 14.7737617, -17.9091129, 15.6998386
1: -4.4685645, 11.6871462, -6.0240598, 15.1364594, -19.6050224, 17.7112064
2: -3.7695451, 13.1007948, -5.0998349, 16.9875908, -20.7571354, 18.2006245
3: -4.4673243, 16.9206219, -6.0953846, 21.7781258, -26.2454491, 23.0160065
4: -3.7639630, 15.6148758, -5.0150137, 20.1749516, -23.9389153, 20.6298904

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5185828, upper bound: 27.5031953
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5185829, upper bound: 27.5136663
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -3.1266623, 11.4615355, -4.6424756, 16.0248470, -19.1515064, 16.1040115
1: -4.4568095, 11.6627026, -6.6280022, 16.3993969, -20.8562069, 18.2907028
2: -3.7594259, 13.0731974, -5.5896873, 18.3416824, -22.1011086, 18.6628838
3: -4.4556742, 16.8855324, -6.6736460, 23.5043488, -27.9600220, 23.5591774
4: -3.7538414, 15.5813236, -5.4507828, 21.7691345, -25.5229759, 21.0321064

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5089277, upper bound: 27.4949283
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5089279, upper bound: 27.4949284
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -4.7071800, 16.1992378, -4.5044422, 15.5995150, -20.3066940, 20.7036762
1: -6.7290516, 16.6196651, -6.4420037, 16.0025711, -22.7316227, 23.0616684
2: -5.6897607, 18.6302376, -5.4514475, 17.9574566, -23.6472168, 24.0816841
3: -6.8060765, 23.8490314, -6.5190701, 22.9807549, -29.7868309, 30.3681011
4: -5.5516272, 22.0954399, -5.3348827, 21.2997704, -26.8513985, 27.4303226

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5299639, upper bound: 27.5178143
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5185829, upper bound: 27.5305277
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -4.6902852, 16.1558075, -4.9108210, 16.8157921, -21.5060768, 21.0666275
1: -6.7054138, 16.5733871, -7.0152240, 17.2347851, -23.9401951, 23.5886116
2: -5.6694956, 18.5777721, -5.9189138, 19.2696590, -24.9391556, 24.4966850
3: -6.7832851, 23.7850208, -7.0767059, 24.6664829, -31.4497681, 30.8617229
4: -5.5334692, 22.0342083, -5.7561159, 22.8612328, -28.3947010, 27.7903252

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5299217, upper bound: 27.5391148
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5299217, upper bound: 27.5409710
time: 0.93 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.61 seconds
IS_A1_B1_A2_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5186280, upper bound: 27.5034970
IS_A1_B1_A2_A1_B1_B2, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5186283, upper bound: 27.5138797
IS_A1_B1_A2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5091115, upper bound: 27.4954060
IS_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5091117, upper bound: 27.5300555
IS_A1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5248261, upper bound: 27.5108817
IS_A1_B1_A2_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5186283, upper bound: 27.5108817
IS_A1_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5256873, upper bound: 27.5313140
IS_A1_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5256873, upper bound: 27.5368391
IS_A1_B2_A1_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5053566, upper bound: 27.5144850
IS_A1_B2_A1_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5053566, upper bound: 27.5136115
IS_A1_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5271045, upper bound: 27.5215295
IS_A1_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5271045, upper bound: 27.5215295
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5217158, upper bound: 27.5048244
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5272600, upper bound: 27.5141705
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5300855, upper bound: 27.5144373
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5380637, upper bound: 27.5268780
IS_A1_B2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.4871268, upper bound: 27.4949605
IS_A1_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5401627, upper bound: 27.5373908
IS_A2_B1_B1_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5144850, upper bound: 27.5122461
IS_A2_B1_B1_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5144850, upper bound: 27.5140094
IS_A2_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.4949605, upper bound: 27.5271045
IS_A2_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.4949605, upper bound: 27.5271045
IS_A2_B1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.4854858, upper bound: 27.4812020
IS_A2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5141707, upper bound: 27.5272604
IS_A2_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5144373, upper bound: 27.5300855
IS_A2_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5268780, upper bound: 27.5380637
IS_A2_B1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.4949605, upper bound: 27.4871268
IS_A2_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5375865, upper bound: 27.5380066
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.4853473, upper bound: 27.4853801
IS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.4853473, upper bound: 27.4853801
IS_A2_B2_A1_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.4856493, upper bound: 27.4990402
IS_A2_B2_A1_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.4978273, upper bound: 27.5042590
IS_A2_B2_A2_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5185828, upper bound: 27.5031953
IS_A2_B2_A2_A1_B1_B2, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5185829, upper bound: 27.5136663
IS_A2_B2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5089277, upper bound: 27.4949283
IS_A2_B2_A2_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5089279, upper bound: 27.4949284
IS_A2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5299639, upper bound: 27.5178143
IS_A2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5185829, upper bound: 27.5305277
IS_A2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5299217, upper bound: 27.5391148
IS_A2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 3, lower bound: -27.5299217, upper bound: 27.5409710

## BFS IS instance: IS_A1_B1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -2.5540318, 9.8917933, -3.8969910, 14.0296011, -16.5836315, 13.7887831
1: -3.7471590, 10.0278521, -5.6356869, 14.3059301, -18.0530891, 15.6635389
2: -3.1771150, 11.2710323, -4.7420278, 16.0134087, -19.1905231, 16.0130596
3: -3.7148616, 14.5879736, -5.6682053, 20.5616055, -24.2764664, 20.2561722
4: -3.2156830, 13.4506903, -4.6631541, 18.9585667, -22.1742496, 18.1138439

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5082593, upper bound: 27.5185173
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5082593, upper bound: 27.5298188
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -4.0725145, 14.3336220, -3.5618644, 12.9171572, -16.9896717, 17.8954811
1: -5.9068017, 14.6883450, -5.1606417, 13.1869688, -19.0937710, 19.8489876
2: -4.9915209, 16.5139713, -4.3412910, 14.8576298, -19.8491516, 20.8552628
3: -5.9278607, 21.1031227, -5.1961222, 19.0373878, -24.9652481, 26.2992439
4: -4.8845754, 19.5199795, -4.2972727, 17.5959225, -22.4804974, 23.8172531

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5136114, upper bound: 27.5053567
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5136114, upper bound: 27.5053567
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -3.8286188, 13.6773224, -4.2670722, 14.9929380, -18.8215542, 17.9443951
1: -5.5450964, 13.9873419, -6.1816254, 15.3483210, -20.8934135, 20.1689663
2: -4.6782236, 15.7261734, -5.2292953, 17.1967506, -21.8749733, 20.9554691
3: -5.5749130, 20.1388111, -6.2076254, 21.9800053, -27.5549183, 26.3464355
4: -4.5991917, 18.5908585, -5.0987215, 20.3454666, -24.9446564, 23.6895790

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4994685, upper bound: 27.4938503
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4994686, upper bound: 27.5219183
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.1501856, 14.7619257, -4.2670722, 14.9929380, -19.1431236, 19.0289974
1: -6.0045290, 15.0758657, -6.1816254, 15.3483210, -21.3528442, 21.2574902
2: -5.0533237, 16.8712635, -5.2292953, 17.1967506, -22.2500744, 22.1005592
3: -6.0436320, 21.6307240, -6.2076254, 21.9800053, -28.0236359, 27.8383484
4: -4.9472756, 19.9544792, -5.0987215, 20.3454666, -25.2927399, 25.0531998

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4994684, upper bound: 27.4993754
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4994686, upper bound: 27.5359812
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -3.8286188, 13.6773224, -4.5044422, 15.5995150, -19.4281330, 18.1817646
1: -5.5450964, 13.9873419, -6.4420037, 16.0025711, -21.5476685, 20.4293404
2: -4.6782236, 15.7261734, -5.4514475, 17.9574566, -22.6356812, 21.1776199
3: -5.5749130, 20.1388111, -6.5190701, 22.9807549, -28.5556679, 26.6578808
4: -4.5991917, 18.5908585, -5.3348827, 21.2997704, -25.8989601, 23.9257412

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4699387, upper bound: 27.4940154
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5032118, upper bound: 27.5201485
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -3.8286188, 13.6773224, -4.9108210, 16.8157921, -20.6444092, 18.5881424
1: -5.5450964, 13.9873419, -7.0152240, 17.2347851, -22.7798805, 21.0025616
2: -4.6782236, 15.7261734, -5.9189138, 19.2696590, -23.9478836, 21.6450882
3: -5.5749130, 20.1388111, -7.0767059, 24.6664829, -30.2413960, 27.2155113
4: -4.5991917, 18.5908585, -5.7561159, 22.8612328, -27.4604225, 24.3469734

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4699387, upper bound: 27.4760454
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5032118, upper bound: 27.5225512
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -2.8249590, 10.7035341, -3.8640900, 13.8447924, -16.6697483, 14.5676231
1: -4.1144567, 10.8470554, -5.5221710, 14.1329212, -18.2473755, 16.3692265
2: -3.5004230, 12.1523218, -4.6442509, 15.8894129, -19.3898354, 16.7965736
3: -4.0744195, 15.6850214, -5.6003876, 20.4276009, -24.5020180, 21.2854061
4: -3.4995542, 14.4455595, -4.6082296, 18.8965511, -22.3961048, 19.0537891

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4865756, upper bound: 27.4857520
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4865756, upper bound: 27.5048244
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2.8249590, 10.7035341, -4.1502252, 14.6663017, -17.4912548, 14.8537598
1: -4.1144567, 10.8470554, -5.9249916, 14.9979420, -19.1123943, 16.7720470
2: -3.5004230, 12.1523218, -5.0062690, 16.8156319, -20.3160553, 17.1585903
3: -4.0744195, 15.6850214, -6.0039525, 21.6142864, -25.6887016, 21.6889744
4: -3.4995542, 14.4455595, -4.9335155, 19.9759998, -23.4755535, 19.3790741

Time for backsubstitution: 2.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4921596, upper bound: 27.4952276
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4871268, upper bound: 27.5141704
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.2554588, 14.9586115, -4.1293182, 14.6288443, -18.8843040, 19.0879269
1: -6.1641321, 15.3125620, -5.9055848, 14.9554243, -21.1195564, 21.2181473
2: -5.2145309, 17.1568089, -4.9716229, 16.8073635, -22.0218945, 22.1284313
3: -6.1905026, 21.9307594, -5.9947290, 21.5765266, -27.7670250, 27.9254875
4: -5.0852590, 20.2982121, -4.9104395, 19.9661102, -25.0513687, 25.2086487

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4922749, upper bound: 27.4893361
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4922749, upper bound: 27.4893361
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.2554588, 14.9586115, -4.4364772, 15.4927216, -19.7481804, 19.3950882
1: -6.1641321, 15.3125620, -6.3414073, 15.8659678, -22.0300999, 21.6539688
2: -5.2145309, 17.1568089, -5.3552303, 17.7862968, -23.0008278, 22.5120373
3: -6.1905026, 21.9307594, -6.4260044, 22.8168812, -29.0073814, 28.3567638
4: -5.0852590, 20.2982121, -5.2513251, 21.1006165, -26.1858749, 25.5495338

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4938503, upper bound: 27.4994687
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4938503, upper bound: 27.4994687
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -4.1501856, 14.7619257, -4.9108210, 16.8157921, -20.9659767, 19.6727467
1: -6.0045290, 15.0758657, -7.0152240, 17.2347851, -23.2393093, 22.0910873
2: -5.0533237, 16.8712635, -5.9189138, 19.2696590, -24.3229828, 22.7901764
3: -6.0436320, 21.6307240, -7.0767059, 24.6664829, -30.7101135, 28.7074261
4: -4.9472756, 19.9544792, -5.7561159, 22.8612328, -27.8085060, 25.7105942

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5032116, upper bound: 27.4995404
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5178081, upper bound: 27.5373859
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -4.5044422, 15.5995150, -3.8286188, 13.6773224, -18.1817646, 19.4281330
1: -6.4420037, 16.0025711, -5.5450964, 13.9873419, -20.4293423, 21.5476685
2: -5.4514475, 17.9574566, -4.6782236, 15.7261734, -21.1776199, 22.6356792
3: -6.5190701, 22.9807549, -5.5749130, 20.1388111, -26.6578808, 28.5556679
4: -5.3348827, 21.2997704, -4.5991917, 18.5908585, -23.9257412, 25.8989601

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_B2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4875275, upper bound: 27.5032115
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4875275, upper bound: 27.4900289
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -4.9108210, 16.8157921, -3.8286188, 13.6773224, -18.5881424, 20.6444092
1: -7.0152240, 17.2347851, -5.5450964, 13.9873419, -21.0025616, 22.7798805
2: -5.9189138, 19.2696590, -4.6782236, 15.7261734, -21.6450882, 23.9478836
3: -7.0767059, 24.6664829, -5.5749130, 20.1388111, -27.2155113, 30.2413960
4: -5.7561159, 22.8612328, -4.5991917, 18.5908585, -24.3469734, 27.4604225

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4875275, upper bound: 27.5032118
time: 0.89 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4875275, upper bound: 27.5287923
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.1502252, 14.6663017, -2.8249590, 10.7035341, -14.8537598, 17.4912548
1: -5.9249916, 14.9979420, -4.1144567, 10.8470554, -16.7720470, 19.1123943
2: -5.0062690, 16.8156319, -3.5004230, 12.1523218, -17.1585903, 20.3160553
3: -6.0039525, 21.6142864, -4.0744195, 15.6850214, -21.6889725, 25.6887016
4: -4.9335155, 19.9759998, -3.4995542, 14.4455595, -19.3790741, 23.4755535

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4952277, upper bound: 27.4921596
time: 1.17 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4952277, upper bound: 27.5272605
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.1293182, 14.6288443, -4.2554588, 14.9586115, -19.0879288, 18.8843040
1: -5.9055848, 14.9554243, -6.1641321, 15.3125620, -21.2181473, 21.1195564
2: -4.9716229, 16.8073635, -5.2145309, 17.1568089, -22.1284313, 22.0218945
3: -5.9947290, 21.5765266, -6.1905026, 21.9307594, -27.9254875, 27.7670269
4: -4.9104395, 19.9661102, -5.0852590, 20.2982121, -25.2086487, 25.0513687

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4893361, upper bound: 27.4922749
time: 0.90 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4893361, upper bound: 27.5300855
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.4364772, 15.4927216, -4.2554588, 14.9586115, -19.3950882, 19.7481804
1: -6.3414073, 15.8659678, -6.1641321, 15.3125620, -21.6539688, 22.0300999
2: -5.3552303, 17.7862968, -5.2145309, 17.1568089, -22.5120392, 23.0008278
3: -6.4260044, 22.8168812, -6.1905026, 21.9307594, -28.3567638, 29.0073776
4: -5.2513251, 21.1006165, -5.0852590, 20.2982121, -25.5495338, 26.1858749

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5000113, upper bound: 27.4980507
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5000113, upper bound: 27.5380638
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -4.9108210, 16.8157921, -4.1501856, 14.7619257, -19.6727467, 20.9659767
1: -7.0152240, 17.2347851, -6.0045290, 15.0758657, -22.0910873, 23.2393093
2: -5.9189138, 19.2696590, -5.0533237, 16.8712635, -22.7901764, 24.3229828
3: -7.0767059, 24.6664829, -6.0436320, 21.6307240, -28.7074261, 30.7101135
4: -5.7561159, 22.8612328, -4.9472756, 19.9544792, -25.7105942, 27.8085060

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A2_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4780672, upper bound: 27.5032116
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4780672, upper bound: 27.5372639
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -4.7071800, 16.1992378, -4.1293182, 14.6288443, -19.3360233, 20.3285542
1: -6.7290516, 16.6196651, -5.9055848, 14.9554243, -21.6844749, 22.5252495
2: -5.6897607, 18.6302376, -4.9716229, 16.8073635, -22.4971237, 23.6018581
3: -6.8060765, 23.8490314, -5.9947290, 21.5765266, -28.3826027, 29.8437614
4: -5.5516272, 22.0954399, -4.9104395, 19.9661102, -25.5177383, 27.0058784

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5202956, upper bound: 27.5158600
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B1_A2

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5202956, upper bound: 27.5158600
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -4.7071800, 16.1992378, -4.4364772, 15.4927216, -20.1999016, 20.6357155
1: -6.7290516, 16.6196651, -6.3414073, 15.8659678, -22.5950203, 22.9610729
2: -5.6897607, 18.6302376, -5.3552303, 17.7862968, -23.4760571, 23.9854660
3: -6.8060765, 23.8490314, -6.4260044, 22.8168812, -29.6229572, 30.2750359
4: -5.5516272, 22.0954399, -5.2513251, 21.1006165, -26.6522446, 27.3467636

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4785638, upper bound: 27.4856566
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4785638, upper bound: 27.5305277
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.4230647, 15.4594355, -4.9108210, 16.8157921, -21.2388554, 20.3702564
1: -6.3247919, 15.8311148, -7.0152240, 17.2347851, -23.5595760, 22.8463364
2: -5.3410764, 17.7477570, -5.9189138, 19.2696590, -24.6107349, 23.6666718
3: -6.4092364, 22.7688160, -7.0767059, 24.6664829, -31.0757198, 29.8455143
4: -5.2383776, 21.0549316, -5.7561159, 22.8612328, -28.0996094, 26.8110466

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5032611, upper bound: 27.4985284
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5032612, upper bound: 27.5365692
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.7579708, 16.5425053, -4.9108210, 16.8157921, -21.5737591, 21.4533253
1: -6.8075414, 16.9199429, -7.0152240, 17.2347851, -24.0423241, 23.9351673
2: -5.7232504, 18.9006500, -5.9189138, 19.2696590, -24.9929085, 24.8195648
3: -6.8910208, 24.2590141, -7.0767059, 24.6664829, -31.5575027, 31.3357143
4: -5.5892906, 22.4202957, -5.7561159, 22.8612328, -28.4505215, 28.1764107

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4675196, upper bound: 27.5026556
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5032612, upper bound: 27.5026556
time: 0.76 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.39 seconds
IS_A1_B1_A2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.5082593, upper bound: 27.5185173
IS_A1_B1_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.5082593, upper bound: 27.5298188
IS_A1_B1_A2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.5136114, upper bound: 27.5053567
IS_A1_B1_A2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.5136114, upper bound: 27.5053567
IS_A1_B1_A2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4994685, upper bound: 27.4938503
IS_A1_B1_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4994686, upper bound: 27.5219183
IS_A1_B1_A2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4994684, upper bound: 27.4993754
IS_A1_B1_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4994686, upper bound: 27.5359812
IS_A1_B2_A1_A2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4699387, upper bound: 27.4940154
IS_A1_B2_A1_A2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.5032118, upper bound: 27.5201485
IS_A1_B2_A1_A2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4699387, upper bound: 27.4760454
IS_A1_B2_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.5032118, upper bound: 27.5225512
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4865756, upper bound: 27.4857520
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4865756, upper bound: 27.5048244
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4921596, upper bound: 27.4952276
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4871268, upper bound: 27.5141704
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4922749, upper bound: 27.4893361
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4922749, upper bound: 27.4893361
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4938503, upper bound: 27.4994687
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4938503, upper bound: 27.4994687
IS_A1_B2_A2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.5032116, upper bound: 27.4995404
IS_A1_B2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.5178081, upper bound: 27.5373859
IS_A2_B1_B1_B2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4875275, upper bound: 27.5032115
IS_A2_B1_B1_B2_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4875275, upper bound: 27.4900289
IS_A2_B1_B1_B2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4875275, upper bound: 27.5032118
IS_A2_B1_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4875275, upper bound: 27.5287923
IS_A2_B1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4952277, upper bound: 27.4921596
IS_A2_B1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4952277, upper bound: 27.5272605
IS_A2_B1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4893361, upper bound: 27.4922749
IS_A2_B1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4893361, upper bound: 27.5300855
IS_A2_B1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.5000113, upper bound: 27.4980507
IS_A2_B1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.5000113, upper bound: 27.5380638
IS_A2_B1_B2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4780672, upper bound: 27.5032116
IS_A2_B1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4780672, upper bound: 27.5372639
IS_A2_B2_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.5202956, upper bound: 27.5158600
IS_A2_B2_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.5202956, upper bound: 27.5158600
IS_A2_B2_A2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4785638, upper bound: 27.4856566
IS_A2_B2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4785638, upper bound: 27.5305277
IS_A2_B2_A2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.5032611, upper bound: 27.4985284
IS_A2_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.5032612, upper bound: 27.5365692
IS_A2_B2_A2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.4675196, upper bound: 27.5026556
IS_A2_B2_A2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 4.39
Output dim: 3, lower bound: -27.5032612, upper bound: 27.5026556

## BFS IS instance: IS_A1_B1_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -2.5540318, 9.8917933, -3.8329315, 13.8454361, -16.3994656, 13.7247248
1: -3.7471590, 10.0278521, -5.5419803, 14.1114769, -17.8586349, 15.5698309
2: -3.1771150, 11.2710323, -4.6620960, 15.7979116, -18.9750233, 15.9331245
3: -3.7148616, 14.5879736, -5.5730829, 20.2939014, -24.0087624, 20.1610489
4: -3.2156830, 13.4506903, -4.5911021, 18.7029591, -21.9186420, 18.0417919

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5212223, upper bound: 27.5273975
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5258249, upper bound: 27.5257343
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5253225, upper bound: 27.5246086
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.8286188, 13.6773224, -4.1616135, 14.7958717, -18.6244907, 17.8389359
1: -5.5450964, 13.9873419, -6.0217419, 15.1112652, -20.6563606, 20.0090790
2: -4.6782236, 15.7261734, -5.0678825, 16.9106216, -21.5888443, 20.7940559
3: -5.5749130, 20.1388111, -6.0606642, 21.6794033, -27.2543163, 26.1994743
4: -4.5991917, 18.5908585, -4.9605665, 20.0012703, -24.6004601, 23.5514259

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_A2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4938345, upper bound: 27.5185003
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_A2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4988576, upper bound: 27.5105873
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4994302, upper bound: 27.4936946
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.1501856, 14.7619257, -4.1616135, 14.7958717, -18.9460564, 18.9235363
1: -6.0045290, 15.0758657, -6.0217419, 15.1112652, -21.1157913, 21.0976048
2: -5.0533237, 16.8712635, -5.0678825, 16.9106216, -21.9639454, 21.9391460
3: -6.0436320, 21.6307240, -6.0606642, 21.6794033, -27.7230339, 27.6913853
4: -4.9472756, 19.9544792, -4.9605665, 20.0012703, -24.9485435, 24.9150467

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5127050, upper bound: 27.4928855
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5150241, upper bound: 27.5251316
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5150241, upper bound: 27.5356484
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -3.8286188, 13.6773224, -4.7756276, 16.5842285, -20.4128456, 18.4529495
1: -5.5450964, 13.9873419, -6.8313746, 16.9627132, -22.5078087, 20.8187141
2: -4.6782236, 15.7261734, -5.7417583, 18.9490433, -23.6272659, 21.4679298
3: -5.5749130, 20.1388111, -6.9129910, 24.3177433, -29.8926563, 27.0517960
4: -4.5991917, 18.5908585, -5.6055784, 22.4766064, -27.0757961, 24.1964340

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4981819, upper bound: 27.5198451
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5031810, upper bound: 27.5163252
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2_B2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5026604, upper bound: 27.5220329
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -4.1501856, 14.7619257, -4.7756276, 16.5842285, -20.7344131, 19.5375519
1: -6.0045290, 15.0758657, -6.8313746, 16.9627132, -22.9672413, 21.9072399
2: -5.0533237, 16.8712635, -5.7417583, 18.9490433, -24.0023670, 22.6130180
3: -6.0436320, 21.6307240, -6.9129910, 24.3177433, -30.3613739, 28.5437088
4: -4.9472756, 19.9544792, -5.6055784, 22.4766064, -27.4238796, 25.5600567

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5155347, upper bound: 27.5319926
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5174808, upper bound: 27.5289618
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5167915, upper bound: 27.5370811
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_B1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -4.7756276, 16.5842285, -3.8286188, 13.6773224, -18.4529495, 20.4128456
1: -6.8313746, 16.9627132, -5.5450964, 13.9873419, -20.8187160, 22.5078087
2: -5.7417583, 18.9490433, -4.6782236, 15.7261734, -21.4679298, 23.6272659
3: -6.9129910, 24.3177433, -5.5749130, 20.1388111, -27.0517960, 29.8926563
4: -5.6055784, 22.4766064, -4.5991917, 18.5908585, -24.1964340, 27.0757961

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_B2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4809362, upper bound: 27.5272715
time: 0.90 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B1_B2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4825270, upper bound: 27.5281769
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4874290, upper bound: 27.4894171
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.1502252, 14.6663017, -2.7469895, 10.5879211, -14.7381458, 17.4132881
1: -5.9249916, 14.9979420, -3.9921873, 10.6958532, -16.6208439, 18.9901276
2: -5.0062690, 16.8156319, -3.3790686, 11.9702339, -16.9765015, 20.1946983
3: -6.0039525, 21.6142864, -3.9698610, 15.5057163, -21.5096684, 25.5841465
4: -4.9335155, 19.9759998, -3.3975906, 14.2315712, -19.1650829, 23.3735867

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4893486, upper bound: 27.4982826
time: 0.79 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4859083, upper bound: 27.5093530
time: 0.88 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4937674, upper bound: 27.4869671
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.1293182, 14.6288443, -4.1501856, 14.7619257, -18.8912411, 18.7790298
1: -5.9055848, 14.9554243, -6.0045290, 15.0758657, -20.9814491, 20.9599533
2: -4.9716229, 16.8073635, -5.0533237, 16.8712635, -21.8428841, 21.8606873
3: -5.9947290, 21.5765266, -6.0436320, 21.6307240, -27.6254539, 27.6201572
4: -4.9104395, 19.9661102, -4.9472756, 19.9544792, -24.8649158, 24.9133835

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4835734, upper bound: 27.5280016
time: 0.84 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4800268, upper bound: 27.5296781
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4890930, upper bound: 27.5297848
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.4364772, 15.4927216, -4.1501856, 14.7619257, -19.1984024, 19.6429062
1: -6.3414073, 15.8659678, -6.0045290, 15.0758657, -21.4172707, 21.8704967
2: -5.3552303, 17.7862968, -5.0533237, 16.8712635, -22.2264900, 22.8396187
3: -6.4260044, 22.8168812, -6.0436320, 21.6307240, -28.0567284, 28.8605118
4: -5.2513251, 21.1006165, -4.9472756, 19.9544792, -25.2058029, 26.0478897

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4947880, upper bound: 27.4901183
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4901094, upper bound: 27.5310407
time: 0.91 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4901094, upper bound: 27.5344999
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -4.7756276, 16.5842285, -4.1501856, 14.7619257, -19.5375519, 20.7344131
1: -6.8313746, 16.9627132, -6.0045290, 15.0758657, -21.9072399, 22.9672413
2: -5.7417583, 18.9490433, -5.0533237, 16.8712635, -22.6130180, 24.0023670
3: -6.9129910, 24.3177433, -6.0436320, 21.6307240, -28.5437088, 30.3613739
4: -5.6055784, 22.4766064, -4.9472756, 19.9544792, -25.5600567, 27.4238796

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B2_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4717809, upper bound: 27.4802174
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4717809, upper bound: 27.5370598
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -4.4230647, 15.4594355, -4.1293182, 14.6288443, -19.0519085, 19.5887527
1: -6.3247919, 15.8311148, -5.9055848, 14.9554243, -21.2802162, 21.7366982
2: -5.3410764, 17.7477570, -4.9716229, 16.8073635, -22.1484375, 22.7193794
3: -6.4092364, 22.7688160, -5.9947290, 21.5765266, -27.9857635, 28.7635441
4: -5.2383776, 21.0549316, -4.9104395, 19.9661102, -25.2044868, 25.9653683

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5120898, upper bound: 27.4990010
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5187892, upper bound: 27.5084611
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5184886, upper bound: 27.5136853
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -4.7579708, 16.5425053, -4.1293182, 14.6288443, -19.3868122, 20.6718216
1: -6.8075414, 16.9199429, -5.9055848, 14.9554243, -21.7629662, 22.8255272
2: -5.7232504, 18.9006500, -4.9716229, 16.8073635, -22.5306129, 23.8722706
3: -6.8910208, 24.2590141, -5.9947290, 21.5765266, -28.4675484, 30.2537422
4: -5.5892906, 22.4202957, -4.9104395, 19.9661102, -25.5553989, 27.3307343

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5120898, upper bound: 27.5022974
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5040295, upper bound: 27.5060664
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4935666, upper bound: 27.4794875
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -4.7579708, 16.5425053, -4.4364772, 15.4927216, -20.2506905, 20.9789829
1: -6.8075414, 16.9199429, -6.3414073, 15.8659678, -22.6735096, 23.2613487
2: -5.7232504, 18.9006500, -5.3552303, 17.7862968, -23.5095444, 24.2558804
3: -6.8910208, 24.2590141, -6.4260044, 22.8168812, -29.7079010, 30.6850185
4: -5.5892906, 22.4202957, -5.2513251, 21.1006165, -26.6899052, 27.6716213

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5218025, upper bound: 27.5151920
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5095363, upper bound: 27.5123561
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.4230647, 15.4594355, -4.7756276, 16.5842285, -21.0072918, 20.2350636
1: -6.3247919, 15.8311148, -6.8313746, 16.9627132, -23.2875061, 22.6624889
2: -5.3410764, 17.7477570, -5.7417583, 18.9490433, -24.2901192, 23.4895134
3: -6.4092364, 22.7688160, -6.9129910, 24.3177433, -30.7269802, 29.6818008
4: -5.2383776, 21.0549316, -5.6055784, 22.4766064, -27.7149849, 26.6605053

Time for backsubstitution: 2.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4991006, upper bound: 27.5348452
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5029177, upper bound: 27.5305055
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5025519, upper bound: 27.5361497
time: 0.83 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 8.11 seconds
IS_A1_B1_A2_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.5258249, upper bound: 27.5257343
IS_A1_B1_A2_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.5253225, upper bound: 27.5246086
IS_A1_B1_A2_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.4988576, upper bound: 27.5105873
IS_A1_B1_A2_A2_B2_A1_B2_B2, status: Status.VERIFIED, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.4994302, upper bound: 27.4936946
IS_A1_B1_A2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.5150241, upper bound: 27.5251316
IS_A1_B1_A2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.5150241, upper bound: 27.5356484
IS_A1_B2_A1_A2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.5031810, upper bound: 27.5163252
IS_A1_B2_A1_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.5026604, upper bound: 27.5220329
IS_A1_B2_A2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.5174808, upper bound: 27.5289618
IS_A1_B2_A2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.5167915, upper bound: 27.5370811
IS_A2_B1_B1_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.4825270, upper bound: 27.5281769
IS_A2_B1_B1_B2_B2_A2_A2_A2, status: Status.VERIFIED, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.4874290, upper bound: 27.4894171
IS_A2_B1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.4859083, upper bound: 27.5093530
IS_A2_B1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.4937674, upper bound: 27.4869671
IS_A2_B1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.4800268, upper bound: 27.5296781
IS_A2_B1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.4890930, upper bound: 27.5297848
IS_A2_B1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.4901094, upper bound: 27.5310407
IS_A2_B1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.4901094, upper bound: 27.5344999
IS_A2_B1_B2_A2_B2_B2_A2_A1, status: Status.VERIFIED, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.4717809, upper bound: 27.4802174
IS_A2_B1_B2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.4717809, upper bound: 27.5370598
IS_A2_B2_A2_A2_B1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.5187892, upper bound: 27.5084611
IS_A2_B2_A2_A2_B1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.5184886, upper bound: 27.5136853
IS_A2_B2_A2_A2_B1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.5040295, upper bound: 27.5060664
IS_A2_B2_A2_A2_B1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.4935666, upper bound: 27.4794875
IS_A2_B2_A2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.5218025, upper bound: 27.5151920
IS_A2_B2_A2_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.5095363, upper bound: 27.5123561
IS_A2_B2_A2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.5029177, upper bound: 27.5305055
IS_A2_B2_A2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 8.11
Output dim: 3, lower bound: -27.5025519, upper bound: 27.5361497

## BFS IS instance: IS_A1_B1_A2_A1_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -2.4913726, 9.7156506, -3.3171334, 12.4178658, -14.9092388, 13.0327835
1: -3.6516457, 9.8432684, -4.7738519, 12.6035995, -16.2552452, 14.6171188
2: -3.0926998, 11.0621529, -3.9715881, 14.1206522, -17.2133522, 15.0337410
3: -3.6226845, 14.3337355, -4.8214316, 18.2380238, -21.8607063, 19.1551666
4: -3.1381807, 13.2036810, -3.9610007, 16.7284031, -19.8665810, 17.1646824

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5180312, upper bound: 27.5232450
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=31.572500228881836
rel_dist={3: [-27.545485351818588, 27.545485351818584]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5400201, upper bound: 27.5388178
time: 0.71 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5420876, upper bound: 27.5420876
time: 0.75 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.67 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 3, lower bound: -27.5400201, upper bound: 27.5388178
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 3, lower bound: -27.5420876, upper bound: 27.5420876

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.1535349, 14.4982834, -4.5284381, 15.6002283, -19.7537613, 19.0267200
1: -6.0371099, 14.8871717, -6.4662418, 16.0064812, -22.0435848, 21.3534126
2: -5.1117692, 16.7495937, -5.4686127, 17.9615231, -23.0732899, 22.2182064
3: -6.0486369, 21.3488541, -6.5212655, 22.9652290, -29.0138664, 27.8701191
4: -4.9890728, 19.7949219, -5.3459496, 21.3026447, -26.2917175, 25.1408710

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5375737, upper bound: 27.5375737
time: 0.74 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5375737, upper bound: 27.5388178
time: 0.66 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -4.8032727, 16.3772182, -4.8523860, 16.5294952, -21.3327637, 21.2296009
1: -6.8679352, 16.8314762, -6.9371209, 16.9844227, -23.8523579, 23.7685966
2: -5.8185549, 18.8867245, -5.8777061, 19.0531940, -24.8717461, 24.7644272
3: -6.9354429, 24.1179543, -7.0046768, 24.3332596, -31.2687035, 31.1226311
4: -5.6641207, 22.3921146, -5.7172041, 22.5852242, -28.2493439, 28.1093178

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5327411, upper bound: 27.5342315
time: 0.95 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5327411, upper bound: 27.5416585
time: 0.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.28 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.28
Output dim: 3, lower bound: -27.5375737, upper bound: 27.5375737
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.28
Output dim: 3, lower bound: -27.5375737, upper bound: 27.5388178
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 4.28
Output dim: 3, lower bound: -27.5327411, upper bound: 27.5342315
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 4.28
Output dim: 3, lower bound: -27.5327411, upper bound: 27.5416585

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -4.1535349, 14.4982834, -4.1535349, 14.4982834, -18.6518173, 18.6518173
1: -6.0371099, 14.8871717, -6.0371099, 14.8871717, -20.9242802, 20.9242802
2: -5.1117692, 16.7495937, -5.1117692, 16.7495937, -21.8613625, 21.8613625
3: -6.0486369, 21.3488541, -6.0486369, 21.3488541, -27.3974915, 27.3974915
4: -4.9890728, 19.7949219, -4.9890728, 19.7949219, -24.7839947, 24.7839947

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5284182, upper bound: 27.5180424
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5356711, upper bound: 27.5356709
time: 0.82 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -4.1535349, 14.4982834, -4.7617588, 16.2741394, -20.4276733, 19.2600422
1: -6.0371099, 14.8871717, -6.8103848, 16.7217350, -22.7588444, 21.6975555
2: -5.1117692, 16.7495937, -5.7682052, 18.7648544, -23.8766232, 22.5177994
3: -6.0486369, 21.3488541, -6.8792419, 23.9701557, -30.0187931, 28.2280960
4: -4.9890728, 19.7949219, -5.6212664, 22.2521553, -27.2412281, 25.4161873

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5180424, upper bound: 27.5284182
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5180424, upper bound: 27.5356711
time: 0.61 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -3.2230899, 11.6287680, -4.2884660, 14.8930588, -18.1161480, 15.9172306
1: -4.5962033, 11.8650751, -6.1348772, 15.2637396, -19.8599415, 17.9999523
2: -3.8960257, 13.3130264, -5.1921759, 17.1302738, -21.0262966, 18.5051994
3: -4.5771689, 17.1330128, -6.1735816, 21.9375877, -26.5147572, 23.3065948
4: -3.8639328, 15.8723202, -5.0859232, 20.3459702, -24.2099018, 20.9582386

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5234048, upper bound: 27.5123845
time: 0.75 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5234048, upper bound: 27.5123845
time: 0.75 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -4.7884359, 16.3409023, -4.8523860, 16.5294952, -21.3179302, 21.1932888
1: -6.8493519, 16.7934227, -6.9371209, 16.9844227, -23.8337727, 23.7305431
2: -5.8027220, 18.8445301, -5.8777061, 19.0531940, -24.8559151, 24.7222347
3: -6.9172134, 24.0655479, -7.0046768, 24.3332596, -31.2504730, 31.0702248
4: -5.6499076, 22.3423691, -5.7172041, 22.5852242, -28.2351322, 28.0595741

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5202348, upper bound: 27.5223131
time: 0.66 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5202348, upper bound: 27.5223131
time: 0.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.19 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 3, lower bound: -27.5284182, upper bound: 27.5180424
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 3, lower bound: -27.5356711, upper bound: 27.5356709
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 3, lower bound: -27.5180424, upper bound: 27.5284182
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 3, lower bound: -27.5180424, upper bound: 27.5356711
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 3, lower bound: -27.5234048, upper bound: 27.5123845
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 3, lower bound: -27.5234048, upper bound: 27.5123845
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 3, lower bound: -27.5202348, upper bound: 27.5223131
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 3, lower bound: -27.5202348, upper bound: 27.5223131

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -4.0778384, 14.2979774, -3.8096104, 13.5872917, -17.6651306, 18.1075859
1: -5.9258318, 14.6709547, -5.5293055, 13.9026499, -19.8284817, 20.2002602
2: -5.0128708, 16.5077305, -4.6612096, 15.6515274, -20.6643982, 21.1689396
3: -5.9401674, 21.0567207, -5.5544548, 20.0211716, -25.9613380, 26.6111755
4: -4.9011049, 19.5171490, -4.5881572, 18.5381012, -23.4392052, 24.1053028

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5185375, upper bound: 27.5036362
time: 0.75 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5185375, upper bound: 27.5167397
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -4.0987930, 14.3596191, -4.0815225, 14.3603363, -18.4591255, 18.4411392
1: -5.9590111, 14.7385178, -5.9201736, 14.7163811, -20.6753902, 20.6586914
2: -5.0401125, 16.5851898, -5.0029421, 16.5449162, -21.5850258, 21.5881290
3: -5.9721651, 21.1508007, -5.9411888, 21.1416397, -27.1138039, 27.0919876
4: -4.9262757, 19.6014023, -4.8949690, 19.5565891, -24.4828644, 24.4963722

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5312954, upper bound: 27.5279474
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5351378, upper bound: 27.5351376
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.8096104, 13.5872917, -4.6626348, 16.0230312, -19.8326416, 18.2499275
1: -5.5293055, 13.9026499, -6.6694798, 16.4517097, -21.9810143, 20.5721302
2: -4.6612096, 15.6515274, -5.6432438, 18.4610100, -23.1222191, 21.2947674
3: -5.5544548, 20.0211716, -6.7423859, 23.6051426, -29.1595974, 26.7635574
4: -4.5881572, 18.5381012, -5.5110698, 21.9012737, -26.4894314, 24.0491714

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4574091, upper bound: 27.4538653
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5120602, upper bound: 27.5120602
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5142513, upper bound: 27.5295046
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.0815225, 14.3603363, -4.7037463, 16.1283112, -20.2098331, 19.0640812
1: -5.9201736, 14.7163811, -6.7285419, 16.5663452, -22.4865189, 21.4449158
2: -5.0029421, 16.5449162, -5.6944818, 18.5913372, -23.5942783, 22.2393990
3: -5.9411888, 21.1416397, -6.7995195, 23.7614555, -29.7026405, 27.9411526
4: -4.8949690, 19.5565891, -5.5563593, 22.0492191, -26.9441872, 25.1129436

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5291501, upper bound: 27.5312954
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5376243, upper bound: 27.5363142
time: 0.84 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -3.2230899, 11.6287680, -3.9908514, 14.1246185, -17.3477077, 15.6196194
1: -4.5962033, 11.8650751, -5.7073898, 14.4452677, -19.0414696, 17.5724640
2: -3.8960257, 13.3130264, -4.8251128, 16.2151756, -20.1111984, 18.1381378
3: -4.5771689, 17.1330128, -5.7597136, 20.8180866, -25.3952541, 22.8927269
4: -3.8639328, 15.8723202, -4.7586651, 19.2707367, -23.1346684, 20.6309814

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5127587, upper bound: 27.4949512
time: 0.81 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5153195, upper bound: 27.5002229
time: 0.90 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -3.1664002, 11.4725723, -4.4219885, 15.3665380, -18.5329380, 15.8945608
1: -4.5174589, 11.7022696, -6.3057327, 15.7019644, -20.2194214, 18.0080013
2: -3.8296685, 13.1277113, -5.3175411, 17.5637760, -21.3934422, 18.4452515
3: -4.4985657, 16.9008827, -6.3416352, 22.5335083, -27.0320721, 23.2425156
4: -3.7975118, 15.6520815, -5.1950760, 20.8472538, -24.6447659, 20.8471565

Time for backsubstitution: 2.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5082467, upper bound: 27.4917604
time: 0.62 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5273898, upper bound: 27.5284934
time: 0.85 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -4.6904068, 16.0923367, -4.4418936, 15.4881372, -20.1785431, 20.5342255
1: -6.7097392, 16.5258865, -6.3562999, 15.8623877, -22.5721188, 22.8821831
2: -5.6791220, 18.5435238, -5.3623009, 17.8052959, -23.4844170, 23.9058247
3: -6.7815223, 23.7042599, -6.4357409, 22.8261089, -29.6076317, 30.1400013
4: -5.5409179, 21.9951706, -5.2633905, 21.1458282, -26.6867466, 27.2585564

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5056268, upper bound: 27.5049727
time: 0.66 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4993632, upper bound: 27.4841682
time: 0.96 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -4.7301283, 16.1942940, -4.7636456, 16.3704605, -21.1005878, 20.9579391
1: -6.7675347, 16.6371593, -6.8063335, 16.7921143, -23.5596485, 23.4434910
2: -5.7288351, 18.6702442, -5.7550316, 18.8174706, -24.5463066, 24.4252758
3: -6.8373637, 23.8555527, -6.8844986, 24.0920811, -30.9294453, 30.7400513
4: -5.5848417, 22.1383724, -5.6106577, 22.3131275, -27.8979683, 27.7490311

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5169379, upper bound: 27.5238022
time: 0.82 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5137724, upper bound: 27.5137722
time: 0.95 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.36 seconds
IS_A1_B1_B1_B1, status: Status.VERIFIED, split count: 4, time: 4.36
Output dim: 3, lower bound: -27.5185375, upper bound: 27.5036362
IS_A1_B1_B1_B2, status: Status.VERIFIED, split count: 4, time: 4.36
Output dim: 3, lower bound: -27.5185375, upper bound: 27.5167397
IS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -27.5312954, upper bound: 27.5279474
IS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -27.5351378, upper bound: 27.5351376
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.36
Output dim: 3, lower bound: -27.5120602, upper bound: 27.5120602
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -27.5142513, upper bound: 27.5295046
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -27.5291501, upper bound: 27.5312954
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -27.5376243, upper bound: 27.5363142
IS_A2_A1_B1_B1, status: Status.VERIFIED, split count: 4, time: 4.36
Output dim: 3, lower bound: -27.5127587, upper bound: 27.4949512
IS_A2_A1_B1_B2, status: Status.VERIFIED, split count: 4, time: 4.36
Output dim: 3, lower bound: -27.5153195, upper bound: 27.5002229
IS_A2_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 4.36
Output dim: 3, lower bound: -27.5082467, upper bound: 27.4917604
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -27.5273898, upper bound: 27.5284934
IS_A2_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 4.36
Output dim: 3, lower bound: -27.5056268, upper bound: 27.5049727
IS_A2_A2_B1_B2, status: Status.VERIFIED, split count: 4, time: 4.36
Output dim: 3, lower bound: -27.4993632, upper bound: 27.4841682
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -27.5169379, upper bound: 27.5238022
IS_A2_A2_B2_B2, status: Status.VERIFIED, split count: 4, time: 4.36
Output dim: 3, lower bound: -27.5137724, upper bound: 27.5137722

## BFS IS instance: IS_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -3.5431085, 12.7629766, -2.5642190, 9.9206409, -13.4637480, 15.3271961
1: -5.1528053, 13.0626659, -3.7614348, 10.0586987, -15.2115030, 16.8240948
2: -4.3640890, 14.7164841, -3.1901212, 11.3052711, -15.6693602, 17.9066010
3: -5.1636481, 18.8170395, -3.7287681, 14.6306629, -19.7943096, 22.5458069
4: -4.3108082, 17.4377174, -3.2276156, 13.4916935, -17.8025017, 20.6653328

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5003675, upper bound: 27.5141920
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5289177, upper bound: 27.5265830
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -4.0987930, 14.3596191, -4.0725145, 14.3336220, -18.4324093, 18.4321308
1: -5.9590111, 14.7385178, -5.9068017, 14.6883450, -20.6473560, 20.6453190
2: -5.0401125, 16.5851898, -4.9915209, 16.5139713, -21.5540829, 21.5767078
3: -5.9721651, 21.1508007, -5.9278607, 21.1031227, -27.0752869, 27.0786610
4: -4.9262757, 19.6014023, -4.8845754, 19.5199795, -24.4462547, 24.4859772

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5220374, upper bound: 27.5281375
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5336223, upper bound: 27.5336224
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.8096104, 13.5872917, -4.6885233, 16.1500626, -19.9596729, 18.2758141
1: -5.5293055, 13.9026499, -6.7007990, 16.5667801, -22.0960846, 20.6034451
2: -4.6612096, 15.6515274, -5.6653900, 18.5710144, -23.2322235, 21.3169155
3: -5.5544548, 20.0211716, -6.7788563, 23.7788734, -29.3333282, 26.8000278
4: -4.5881572, 18.5381012, -5.5313053, 22.0295525, -26.6177082, 24.0694065

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4352586, upper bound: 27.4221604
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5035492, upper bound: 27.5260189
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5077775, upper bound: 27.5092174
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -2.5642190, 9.9206409, -4.1479659, 14.5163727, -17.0805912, 14.0686073
1: -3.7614348, 10.0586987, -5.9375362, 14.8708973, -18.6323280, 15.9962349
2: -3.1901212, 11.3052711, -5.0185385, 16.6945400, -19.8846569, 16.3238087
3: -3.7287681, 14.6306629, -5.9801102, 21.4005146, -25.1292820, 20.6107731
4: -3.2276156, 13.4916935, -4.9332118, 19.8415394, -23.0691547, 18.4249058

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5141920, upper bound: 27.5003781
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5277313, upper bound: 27.5289177
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -4.0725145, 14.3336220, -4.7037463, 16.1283112, -20.2008247, 19.0373631
1: -5.9068017, 14.6883450, -6.7285419, 16.5663452, -22.4731445, 21.4168873
2: -4.9915209, 16.5139713, -5.6944818, 18.5913372, -23.5828590, 22.2084541
3: -5.9278607, 21.1031227, -6.7995195, 23.7614555, -29.6893158, 27.9026413
4: -4.8845754, 19.5199795, -5.5563593, 22.0492191, -26.9337921, 25.0763359

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5328554, upper bound: 27.5229522
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5360959, upper bound: 27.5343425
time: 1.01 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -3.1014481, 11.3098688, -4.2969189, 15.1604424, -18.2618904, 15.6067839
1: -4.4257274, 11.5272837, -6.1377296, 15.4557438, -19.8814678, 17.6650124
2: -3.7457967, 12.9336891, -5.1518221, 17.2759323, -21.0217247, 18.0855103
3: -4.4112682, 16.6676140, -6.1880183, 22.2254620, -26.6367302, 22.8556290
4: -3.7250721, 15.4268341, -5.0562973, 20.5129490, -24.2380199, 20.4831295

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4813156, upper bound: 27.5100682
time: 0.74 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4813156, upper bound: 27.4819699
time: 0.77 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -4.7301283, 16.1942940, -4.5608373, 15.7864647, -20.5165939, 20.7551308
1: -6.7675347, 16.6371593, -6.5265927, 16.1868019, -22.9543343, 23.1637516
2: -5.7288351, 18.6702442, -5.5206504, 18.1384907, -23.8673248, 24.1908894
3: -6.8373637, 23.8555527, -6.6047268, 23.2501583, -30.0875225, 30.4602795
4: -5.5848417, 22.1383724, -5.3984323, 21.5136871, -27.0985279, 27.5368042

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5126208, upper bound: 27.5199330
time: 0.81 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5126994, upper bound: 27.5197971
time: 0.74 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.26 seconds
IS_A1_B1_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5003675, upper bound: 27.5141920
IS_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5289177, upper bound: 27.5265830
IS_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5220374, upper bound: 27.5281375
IS_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5336223, upper bound: 27.5336224
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5035492, upper bound: 27.5260189
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5077775, upper bound: 27.5092174
IS_A1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5141920, upper bound: 27.5003781
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5277313, upper bound: 27.5289177
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5328554, upper bound: 27.5229522
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5360959, upper bound: 27.5343425
IS_A2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.4813156, upper bound: 27.5100682
IS_A2_A1_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.4813156, upper bound: 27.4819699
IS_A2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5126208, upper bound: 27.5199330
IS_A2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5126994, upper bound: 27.5197971

## BFS IS instance: IS_A1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -3.6752939, 13.2793655, -2.5022404, 9.7450447, -13.4203377, 15.7816038
1: -5.3211675, 13.5455418, -3.6744237, 9.8711271, -15.1922932, 17.2199650
2: -4.4957652, 15.1806536, -3.1111042, 11.0970707, -15.5928345, 18.2917576
3: -5.3290815, 19.4819183, -3.6440992, 14.3708162, -19.6998959, 23.1260185
4: -4.4302101, 18.0041199, -3.1551530, 13.2421923, -17.6723995, 21.1592731

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4726562, upper bound: 27.5066576
time: 0.84 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4726562, upper bound: 27.5265830
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -3.8418636, 13.6622763, -4.0725145, 14.3336220, -18.1754837, 17.7347870
1: -5.5766206, 13.9943857, -5.9068017, 14.6883450, -20.2649651, 19.9011879
2: -4.7086067, 15.7499847, -4.9915209, 16.5139713, -21.2225780, 20.7415028
3: -5.6003809, 20.1255226, -5.9278607, 21.1031227, -26.7035027, 26.0533829
4: -4.6246991, 18.6176338, -4.8845754, 19.5199795, -24.1446781, 23.5022087

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4808382, upper bound: 27.5199708
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4808382, upper bound: 27.5168415
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -4.2110014, 14.8474216, -3.9807291, 14.0967293, -18.3077316, 18.8281479
1: -6.1010957, 15.1927423, -5.7785311, 14.4353905, -20.5364838, 20.9712734
2: -5.1551867, 17.0240612, -4.8804145, 16.2275200, -21.3827038, 21.9044704
3: -6.1287923, 21.7718430, -5.8039927, 20.7542324, -26.8830242, 27.5758324
4: -5.0334172, 20.1421394, -4.7843266, 19.1849289, -24.2183456, 24.9264660

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5319733, upper bound: 27.5323215
time: 0.84 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5319732, upper bound: 27.5319733
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.3817334, 12.4283094, -4.5076818, 15.6601610, -19.0418911, 16.9359913
1: -4.9561958, 12.6648188, -6.4587851, 16.0536194, -21.0098133, 19.1236038
2: -4.1724143, 14.2834492, -5.4607258, 17.9999695, -22.1723843, 19.7441730
3: -4.9773903, 18.3161221, -6.5358014, 23.0667477, -28.0441380, 24.8519230
4: -4.1479683, 16.9420795, -5.3471928, 21.3632507, -25.5112171, 22.2892704

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5016380, upper bound: 27.5192707
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4913369, upper bound: 27.5096670
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -2.5022404, 9.7450447, -4.2932420, 15.0226088, -17.5248489, 14.0382862
1: -3.6744237, 9.8711271, -6.1268425, 15.3433542, -19.0177784, 15.9979696
2: -3.1111042, 11.0970707, -5.1593552, 17.1682262, -20.2793312, 16.2564259
3: -3.6440992, 14.3708162, -6.1663704, 22.0417099, -25.6858082, 20.5371819
4: -3.1551530, 13.2421923, -5.0557814, 20.3879566, -23.5431080, 18.2979717

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5085515, upper bound: 27.4928875
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5085519, upper bound: 27.5277188
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -4.0725145, 14.3336220, -4.4157543, 15.3745213, -19.4470348, 18.7493744
1: -5.9068017, 14.6883450, -6.3180275, 15.7630386, -21.6698399, 21.0063725
2: -4.9915209, 16.5139713, -5.3406258, 17.6903515, -22.6818676, 21.8545971
3: -5.9278607, 21.1031227, -6.3974695, 22.6583118, -28.5861721, 27.5005913
4: -4.8845754, 19.5199795, -5.2376380, 20.9894657, -25.8740406, 24.7576160

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5249377, upper bound: 27.5096907
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5249379, upper bound: 27.5096907
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -3.9807291, 14.0967293, -4.8190136, 16.5898056, -20.5705318, 18.9157429
1: -5.7785311, 14.4353905, -6.8869362, 16.9943218, -22.7728539, 21.3223228
2: -4.8804145, 16.2275200, -5.8054390, 19.0011711, -23.8815804, 22.0329571
3: -5.8039927, 20.7542324, -6.9535165, 24.3422356, -30.1462288, 27.7077465
4: -4.7843266, 19.1849289, -5.6574912, 22.5475235, -27.3318501, 24.8424206

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5348130, upper bound: 27.5321940
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5348030, upper bound: 27.5328253
time: 0.91 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.38 seconds
IS_A1_B1_B2_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 4.38
Output dim: 3, lower bound: -27.4726562, upper bound: 27.5066576
IS_A1_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -27.4726562, upper bound: 27.5265830
IS_A1_B1_B2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 4.38
Output dim: 3, lower bound: -27.4808382, upper bound: 27.5199708
IS_A1_B1_B2_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 4.38
Output dim: 3, lower bound: -27.4808382, upper bound: 27.5168415
IS_A1_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -27.5319733, upper bound: 27.5323215
IS_A1_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -27.5319732, upper bound: 27.5319733
IS_A1_B2_A1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 4.38
Output dim: 3, lower bound: -27.5016380, upper bound: 27.5192707
IS_A1_B2_A1_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 4.38
Output dim: 3, lower bound: -27.4913369, upper bound: 27.5096670
IS_A1_B2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 4.38
Output dim: 3, lower bound: -27.5085515, upper bound: 27.4928875
IS_A1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -27.5085519, upper bound: 27.5277188
IS_A1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -27.5249377, upper bound: 27.5096907
IS_A1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -27.5249379, upper bound: 27.5096907
IS_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -27.5348130, upper bound: 27.5321940
IS_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 3, lower bound: -27.5348030, upper bound: 27.5328253

## BFS IS instance: IS_A1_B1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -3.6163716, 13.1977367, -2.5022404, 9.7450447, -13.3614159, 15.6999769
1: -5.2282772, 13.4339600, -3.6744237, 9.8711271, -15.0994024, 17.1083832
2: -4.3980064, 15.0428381, -3.1111042, 11.0970707, -15.4950771, 18.1539402
3: -5.2449408, 19.3518124, -3.6440992, 14.3708162, -19.6157532, 22.9959087
4: -4.3473883, 17.8298836, -3.1551530, 13.2421923, -17.5895805, 20.9850349

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4630720, upper bound: 27.4886229
time: 0.72 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4725649, upper bound: 27.4720204
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -3.7961764, 13.6746712, -3.8166499, 13.6553154, -17.4514885, 17.4913216
1: -5.5310402, 13.9528542, -5.5604353, 13.9719915, -19.5030270, 19.5132885
2: -4.6830511, 15.6470499, -4.6935310, 15.7154608, -20.3985119, 20.3405781
3: -5.5528083, 20.0526142, -5.5860801, 20.1099339, -25.6627426, 25.6386929
4: -4.6056976, 18.5288219, -4.6148157, 18.5854836, -23.1911812, 23.1436386

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5319732, upper bound: 27.5319733
time: 0.77 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5319732, upper bound: 27.5319733
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -4.1444931, 14.6573362, -3.9503973, 14.0103207, -18.1548100, 18.6077328
1: -6.0061307, 14.9935369, -5.7351961, 14.3447037, -20.3508339, 20.7287331
2: -5.0737710, 16.8021221, -4.8431430, 16.1266575, -21.2004280, 21.6452637
3: -6.0325556, 21.4929428, -5.7602296, 20.6274586, -26.6600151, 27.2531719
4: -4.9587917, 19.8774223, -4.7503166, 19.0641060, -24.0228958, 24.6277390

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_A2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4595305, upper bound: 27.4565515
time: 0.80 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4412493, upper bound: 27.4412492
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -2.5022404, 9.7450447, -4.2479081, 15.0058718, -17.5081120, 13.9929523
1: -3.6744237, 9.8711271, -6.0685949, 15.3004951, -18.9749184, 15.9397221
2: -3.1111042, 11.0970707, -5.0917230, 17.1055222, -20.2166252, 16.1887932
3: -3.6440992, 14.3708162, -6.1180878, 22.0045414, -25.6486397, 20.4888992
4: -3.1551530, 13.2421923, -5.0015993, 20.3164043, -23.4715557, 18.2437897

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5038994, upper bound: 27.5211471
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5082468, upper bound: 27.5274903
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -4.0725145, 14.3336220, -4.1017227, 14.5571432, -18.6296577, 18.4353409
1: -5.9068017, 14.6883450, -5.8667536, 14.8792152, -20.7860165, 20.5550976
2: -4.9915209, 16.5139713, -4.9373026, 16.7234840, -21.7150040, 21.4512749
3: -5.9278607, 21.1031227, -5.9568057, 21.4734745, -27.4013348, 27.0599289
4: -4.8845754, 19.5199795, -4.8808289, 19.8692608, -24.7538357, 24.4008083

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4723881, upper bound: 27.5012524
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5142520, upper bound: 27.5096907
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -4.0725145, 14.3336220, -4.4147778, 15.4354153, -19.5079269, 18.7483978
1: -5.9068017, 14.6883450, -6.3109970, 15.8050585, -21.7118607, 20.9993382
2: -4.9915209, 16.5139713, -5.3295865, 17.7188568, -22.7103767, 21.8435574
3: -5.9278607, 21.1031227, -6.3956242, 22.7347984, -28.6626587, 27.4987469
4: -4.8845754, 19.5199795, -5.2289000, 21.0236416, -25.9082165, 24.7488766

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4723881, upper bound: 27.4761048
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4723881, upper bound: 27.5163759
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -3.8166499, 13.6553154, -4.3584652, 15.3312187, -19.1478672, 18.0137787
1: -5.5604353, 13.9719915, -6.2752800, 15.6623421, -21.2227745, 20.2472668
2: -4.6935310, 15.7154608, -5.2757173, 17.5295658, -22.2230968, 20.9911747
3: -5.5860801, 20.1099339, -6.3255396, 22.4986725, -28.0847530, 26.4354744
4: -4.6148157, 18.5854836, -5.1822186, 20.8154888, -25.4303055, 23.7677021

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5107466, upper bound: 27.4890164
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5107469, upper bound: 27.5321940
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -3.9503973, 14.0103207, -4.7126598, 16.3217068, -20.2721004, 18.7229767
1: -5.7351961, 14.3447037, -6.7439966, 16.7146778, -22.4498730, 21.0886974
2: -4.8431430, 16.1266575, -5.6869459, 18.6883106, -23.5314541, 21.8136024
3: -5.7602296, 20.6274586, -6.8129230, 23.9549770, -29.7152061, 27.4403820
4: -4.7503166, 19.0641060, -5.5531120, 22.1783371, -26.9286537, 24.6172161

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5347440, upper bound: 27.5328253
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5347441, upper bound: 27.5328253
time: 0.75 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.96 seconds
IS_A1_B1_B2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 4.96
Output dim: 3, lower bound: -27.4630720, upper bound: 27.4886229
IS_A1_B1_B2_B1_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 4.96
Output dim: 3, lower bound: -27.4725649, upper bound: 27.4720204
IS_A1_B1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.96
Output dim: 3, lower bound: -27.5319732, upper bound: 27.5319733
IS_A1_B1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.96
Output dim: 3, lower bound: -27.5319732, upper bound: 27.5319733
IS_A1_B1_B2_B2_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 4.96
Output dim: 3, lower bound: -27.4595305, upper bound: 27.4565515
IS_A1_B1_B2_B2_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 4.96
Output dim: 3, lower bound: -27.4412493, upper bound: 27.4412492
IS_A1_B2_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.96
Output dim: 3, lower bound: -27.5038994, upper bound: 27.5211471
IS_A1_B2_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.96
Output dim: 3, lower bound: -27.5082468, upper bound: 27.5274903
IS_A1_B2_A2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.96
Output dim: 3, lower bound: -27.4723881, upper bound: 27.5012524
IS_A1_B2_A2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.96
Output dim: 3, lower bound: -27.5142520, upper bound: 27.5096907
IS_A1_B2_A2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.96
Output dim: 3, lower bound: -27.4723881, upper bound: 27.4761048
IS_A1_B2_A2_A2_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.96
Output dim: 3, lower bound: -27.4723881, upper bound: 27.5163759
IS_A1_B2_A2_A2_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.96
Output dim: 3, lower bound: -27.5107466, upper bound: 27.4890164
IS_A1_B2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.96
Output dim: 3, lower bound: -27.5107469, upper bound: 27.5321940
IS_A1_B2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.96
Output dim: 3, lower bound: -27.5347440, upper bound: 27.5328253
IS_A1_B2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.96
Output dim: 3, lower bound: -27.5347441, upper bound: 27.5328253

## BFS IS instance: IS_A1_B1_B2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -3.7961764, 13.6746712, -3.5607264, 12.9416990, -16.7378750, 17.2353973
1: -5.5310402, 13.9528542, -5.2141333, 13.2051258, -18.7361622, 19.1669865
2: -4.6830511, 15.6470499, -4.4066572, 14.8653774, -19.5484276, 20.0537071
3: -5.5528083, 20.0526142, -5.2344880, 19.0509129, -24.6037197, 25.2870998
4: -4.6056976, 18.5288219, -4.3556018, 17.5894203, -22.1951180, 22.8844242

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5266169, upper bound: 27.5134475
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5266173, upper bound: 27.5134475
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -3.7961764, 13.6746712, -3.9117620, 13.9007854, -17.6969585, 17.5864334
1: -5.5310402, 13.9528542, -5.6800594, 14.2296906, -19.7607269, 19.6329136
2: -4.6830511, 15.6470499, -4.7957854, 15.9988871, -20.6819324, 20.4428349
3: -5.5528083, 20.0526142, -5.7044764, 20.4669704, -26.0197773, 25.7570877
4: -4.6056976, 18.5288219, -4.7071714, 18.9113293, -23.5170269, 23.2359924

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5266169, upper bound: 27.5203753
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5266173, upper bound: 27.5134475
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -2.4554131, 9.6039133, -4.2559009, 15.0541430, -17.5095520, 13.8598127
1: -3.6096647, 9.7242384, -6.0739379, 15.3389006, -18.9485607, 15.7981758
2: -3.0548036, 10.9353580, -5.0866423, 17.1446381, -20.1994400, 16.0219994
3: -3.5781159, 14.1650181, -6.1219020, 22.0596714, -25.6377850, 20.2869205
4: -3.1037123, 13.0521202, -4.9941101, 20.3793583, -23.4830685, 18.0462303

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5215428, upper bound: 27.5207325
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5269338, upper bound: 27.5200437
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -2.4907243, 9.7118692, -4.1650853, 14.7730980, -17.2638226, 13.8769550
1: -3.6570697, 9.8358288, -5.9507952, 15.0554085, -18.7124786, 15.7866211
2: -3.0968137, 11.0572834, -4.9895597, 16.8335609, -19.9303722, 16.0468407
3: -3.6264253, 14.3225088, -5.9994764, 21.6651878, -25.2916126, 20.3219833
4: -3.1422021, 13.1959810, -4.9098778, 19.9947853, -23.1369858, 18.1058578

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5217764, upper bound: 27.5268850
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5273571, upper bound: 27.5274866
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -3.8166499, 13.6553154, -4.2827721, 15.2352228, -19.0518723, 17.9380817
1: -5.5604353, 13.9719915, -6.1693459, 15.5352964, -21.0957279, 20.1413345
2: -4.6935310, 15.7154608, -5.1690807, 17.3722725, -22.0658035, 20.8845387
3: -5.5860801, 20.1099339, -6.2352834, 22.3420296, -27.9281063, 26.3452168
4: -4.6148157, 18.5854836, -5.0922117, 20.6209373, -25.2357521, 23.6776962

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4836543, upper bound: 27.4788587
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4836543, upper bound: 27.5088691
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -3.5607264, 12.9416990, -4.7126598, 16.3217068, -19.8824329, 17.6543579
1: -5.2141333, 13.2051258, -6.7439966, 16.7146778, -21.9288101, 19.9491196
2: -4.4066572, 14.8653774, -5.6869459, 18.6883106, -23.0949669, 20.5523224
3: -5.2344880, 19.0509129, -6.8129230, 23.9549770, -29.1894646, 25.8638363
4: -4.3556018, 17.5894203, -5.5531120, 22.1783371, -26.5339394, 23.1425323

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5136535, upper bound: 27.5272729
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5136535, upper bound: 27.5167603
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -3.9117620, 13.9007854, -4.7126598, 16.3217068, -20.2334690, 18.6134434
1: -5.6800594, 14.2296906, -6.7439966, 16.7146778, -22.3947353, 20.9736824
2: -4.7957854, 15.9988871, -5.6869459, 18.6883106, -23.4840965, 21.6858292
3: -5.7044764, 20.4669704, -6.8129230, 23.9549770, -29.6594543, 27.2798920
4: -4.7071714, 18.9113293, -5.5531120, 22.1783371, -26.8855057, 24.4644394

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5347404, upper bound: 27.5284833
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5347427, upper bound: 27.5322651
time: 0.91 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 7.62 seconds
IS_A1_B1_B2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 7.62
Output dim: 3, lower bound: -27.5266169, upper bound: 27.5134475
IS_A1_B1_B2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 7.62
Output dim: 3, lower bound: -27.5266173, upper bound: 27.5134475
IS_A1_B1_B2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 7.62
Output dim: 3, lower bound: -27.5266169, upper bound: 27.5203753
IS_A1_B1_B2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 7.62
Output dim: 3, lower bound: -27.5266173, upper bound: 27.5134475
IS_A1_B2_A2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 7.62
Output dim: 3, lower bound: -27.5215428, upper bound: 27.5207325
IS_A1_B2_A2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 7.62
Output dim: 3, lower bound: -27.5269338, upper bound: 27.5200437
IS_A1_B2_A2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 7.62
Output dim: 3, lower bound: -27.5217764, upper bound: 27.5268850
IS_A1_B2_A2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 7.62
Output dim: 3, lower bound: -27.5273571, upper bound: 27.5274866
IS_A1_B2_A2_A2_B2_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 7.62
Output dim: 3, lower bound: -27.4836543, upper bound: 27.4788587
IS_A1_B2_A2_A2_B2_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 7.62
Output dim: 3, lower bound: -27.4836543, upper bound: 27.5088691
IS_A1_B2_A2_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 7.62
Output dim: 3, lower bound: -27.5136535, upper bound: 27.5272729
IS_A1_B2_A2_A2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 8, time: 7.62
Output dim: 3, lower bound: -27.5136535, upper bound: 27.5167603
IS_A1_B2_A2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 7.62
Output dim: 3, lower bound: -27.5347404, upper bound: 27.5284833
IS_A1_B2_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 7.62
Output dim: 3, lower bound: -27.5347427, upper bound: 27.5322651

## BFS IS instance: IS_A1_B1_B2_B2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -3.7961764, 13.6746712, -3.4136868, 12.5346861, -16.3308620, 17.0883579
1: -5.5310402, 13.9528542, -4.9875507, 12.7703829, -18.3014221, 18.9404049
2: -4.6830511, 15.6470499, -4.2113910, 14.3788033, -19.0618553, 19.8584404
3: -5.5528083, 20.0526142, -5.0113583, 18.4539299, -24.0067368, 25.0639706
4: -4.6056976, 18.5288219, -4.1760516, 17.0105629, -21.6162605, 22.7048740

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4698325, upper bound: 27.4821676
time: 0.90 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4698325, upper bound: 27.5134475
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -3.7961764, 13.6746712, -3.7418022, 13.6122818, -17.4084568, 17.4164715
1: -5.5310402, 13.9528542, -5.4465876, 13.8609514, -19.3919888, 19.3994408
2: -4.6830511, 15.6470499, -4.5936899, 15.5233393, -20.2063904, 20.2407398
3: -5.5528083, 20.0526142, -5.4781079, 19.9450760, -25.4978828, 25.5307198
4: -4.6056976, 18.5288219, -4.5295615, 18.3716660, -22.9773636, 23.0583839

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4698325, upper bound: 27.4821677
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4698325, upper bound: 27.5318900
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -3.7961764, 13.6746712, -3.7598753, 13.4838772, -17.2800503, 17.4345436
1: -5.5310402, 13.9528542, -5.4471669, 13.7843618, -19.3154030, 19.4000206
2: -4.6830511, 15.6470499, -4.5944033, 15.5012827, -20.1843338, 20.2414532
3: -5.5528083, 20.0526142, -5.4762363, 19.8563271, -25.4091301, 25.5288486
4: -4.6056976, 18.5288219, -4.5234141, 18.3234520, -22.9291496, 23.0522366

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5266674, upper bound: 27.5151162
time: 0.84 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5266674, upper bound: 27.5203713
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -3.7961764, 13.6746712, -4.0830555, 14.5707026, -18.3668785, 17.7577248
1: -5.5310402, 13.9528542, -5.9086103, 14.8755217, -20.4065628, 19.8614597
2: -4.6830511, 15.6470499, -4.9708381, 16.6487923, -21.3318443, 20.6178856
3: -5.5528083, 20.0526142, -5.9462061, 21.3504848, -26.9032898, 25.9988194
4: -4.6056976, 18.5288219, -4.8718505, 19.6879902, -24.2936878, 23.4006729

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5266677, upper bound: 27.5284284
time: 0.88 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5266677, upper bound: 27.5203713
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -2.3943224, 9.3895922, -4.2488623, 15.0343409, -17.4286633, 13.6384544
1: -3.5192661, 9.5040503, -6.0641527, 15.3180656, -18.8373318, 15.5682030
2: -2.9814889, 10.6854925, -5.0784311, 17.1216755, -20.1031647, 15.7639236
3: -3.4865637, 13.8462811, -6.1119714, 22.0311699, -25.5177345, 19.9582500
4: -3.0347447, 12.7538433, -4.9867449, 20.3524876, -23.3872318, 17.7405872

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5215428, upper bound: 27.5200173
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5215428, upper bound: 27.5200437
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -2.4119775, 9.4852886, -4.2447519, 15.0242281, -17.4362030, 13.7300406
1: -3.5455871, 9.5994682, -6.0582247, 15.3075581, -18.8531456, 15.6576929
2: -2.9991684, 10.7955561, -5.0730500, 17.1099434, -20.1091118, 15.8686047
3: -3.5147610, 13.9946690, -6.1061897, 22.0168171, -25.5315781, 20.1008587
4: -3.0533216, 12.8873463, -4.9818749, 20.3383522, -23.3916721, 17.8692207

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5205584, upper bound: 27.5190317
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5205583, upper bound: 27.5190317
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -2.4308877, 9.5026283, -4.1583772, 14.7534924, -17.1843796, 13.6610041
1: -3.5684953, 9.6198235, -5.9413776, 15.0348978, -18.6033936, 15.5612011
2: -3.0251880, 10.8134842, -4.9816260, 16.8107986, -19.8359871, 15.7951097
3: -3.5367579, 14.0115194, -5.9898686, 21.6368561, -25.1736145, 20.0013847
4: -3.0747523, 12.9050751, -4.9027109, 19.9681072, -23.0428600, 17.8077850

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5168083, upper bound: 27.5223826
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5194416, upper bound: 27.5229933
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -2.4477310, 9.5949354, -4.1541090, 14.7437477, -17.1914787, 13.7490444
1: -3.5937524, 9.7128859, -5.9352255, 15.0245523, -18.6183014, 15.6481113
2: -3.0417714, 10.9195328, -4.9761701, 16.7994766, -19.8412476, 15.8957024
3: -3.5639639, 14.1547604, -5.9835706, 21.6229916, -25.1869545, 20.1383305
4: -3.0924377, 13.0336895, -4.8978262, 19.9545536, -23.0469875, 17.9315147

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5226989, upper bound: 27.5226129
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5247743, upper bound: 27.5232703
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -3.4136868, 12.5346861, -4.7126598, 16.3217068, -19.7353935, 17.2473450
1: -4.9875507, 12.7703829, -6.7439966, 16.7146778, -21.7022247, 19.5143795
2: -4.2113910, 14.3788033, -5.6869459, 18.6883106, -22.8997021, 20.0657501
3: -5.0113583, 18.4539299, -6.8129230, 23.9549770, -28.9663353, 25.2668514
4: -4.1760516, 17.0105629, -5.5531120, 22.1783371, -26.3543892, 22.5636711

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5118013, upper bound: 27.5232567
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5083168, upper bound: 27.5272648
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -3.9049628, 13.8808050, -4.6371427, 16.0799084, -19.9848709, 18.5179482
1: -5.6701317, 14.2086916, -6.6357341, 16.4686508, -22.1387825, 20.8444214
2: -4.7874732, 15.9757004, -5.6012249, 18.4092083, -23.1966801, 21.5769253
3: -5.6944275, 20.4382610, -6.7015224, 23.6026440, -29.2970695, 27.1397839
4: -4.6996660, 18.8837395, -5.4738159, 21.8496933, -26.5493584, 24.3575554

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5192100, upper bound: 27.5233307
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5192100, upper bound: 27.5284833
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -3.9030447, 13.8769941, -4.6650405, 16.1955986, -20.0986443, 18.5420341
1: -5.6671729, 14.2046556, -6.6755085, 16.5827160, -22.2498875, 20.8801651
2: -4.7846470, 15.9713144, -5.6286211, 18.5420589, -23.3267059, 21.5999317
3: -5.6916494, 20.4330063, -6.7462692, 23.7741299, -29.4657784, 27.1792755
4: -4.6972704, 18.8785801, -5.5010204, 22.0045643, -26.7018318, 24.3796005

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4936556, upper bound: 27.4854784
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5192573, upper bound: 27.5269884
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5192574, upper bound: 27.5317091
time: 0.73 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 7.04 seconds
IS_A1_B1_B2_B2_A2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.4698325, upper bound: 27.4821676
IS_A1_B1_B2_B2_A2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.4698325, upper bound: 27.5134475
IS_A1_B1_B2_B2_A2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.4698325, upper bound: 27.4821677
IS_A1_B1_B2_B2_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.4698325, upper bound: 27.5318900
IS_A1_B1_B2_B2_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.5266674, upper bound: 27.5151162
IS_A1_B1_B2_B2_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.5266674, upper bound: 27.5203713
IS_A1_B1_B2_B2_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.5266677, upper bound: 27.5284284
IS_A1_B1_B2_B2_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.5266677, upper bound: 27.5203713
IS_A1_B2_A2_A1_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.5215428, upper bound: 27.5200173
IS_A1_B2_A2_A1_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.5215428, upper bound: 27.5200437
IS_A1_B2_A2_A1_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.5205584, upper bound: 27.5190317
IS_A1_B2_A2_A1_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.5205583, upper bound: 27.5190317
IS_A1_B2_A2_A1_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.5168083, upper bound: 27.5223826
IS_A1_B2_A2_A1_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.5194416, upper bound: 27.5229933
IS_A1_B2_A2_A1_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.5226989, upper bound: 27.5226129
IS_A1_B2_A2_A1_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.5247743, upper bound: 27.5232703
IS_A1_B2_A2_A2_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.5118013, upper bound: 27.5232567
IS_A1_B2_A2_A2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.5083168, upper bound: 27.5272648
IS_A1_B2_A2_A2_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.5192100, upper bound: 27.5233307
IS_A1_B2_A2_A2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.5192100, upper bound: 27.5284833
IS_A1_B2_A2_A2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.5192573, upper bound: 27.5269884
IS_A1_B2_A2_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 7.04
Output dim: 3, lower bound: -27.5192574, upper bound: 27.5317091
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=31.572500228881836
rel_dist={3: [-27.542158575057197, 27.5421585750572]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1124.33 seconds
