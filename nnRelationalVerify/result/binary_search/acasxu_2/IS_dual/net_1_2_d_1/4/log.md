## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_2.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 47.0393777385


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003)
1: (-10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643)
2: (-10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861)
3: (-15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287)
4: (-17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024)

## BASE Result
execution time: IAR + LP analysis = 1.66 + 1.92 = 3.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -47.1809221, upper bound: 47.1809221


# Binary Search by BASE starts (time budget: 1196.42 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1666667, mid=0.1666667, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 1) starts
Candidate diff: 0.0833333


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0833333, mid=0.0833333, abs_max=50.8192024230957
rel_dist={4: [-47.18088696914194, 47.18088696914194]}

## Binary search (step 2) starts
Candidate diff: 0.0416667


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0416667, mid=0.0416667, abs_max=50.8192024230957
rel_dist={4: [-47.180735877080295, 47.18073587708028]}

## Binary search (step 3) starts
Candidate diff: 0.0208333


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0208333, mid=0.0208333, abs_max=50.8192024230957
rel_dist={4: [-47.18036906270342, 47.18036906270342]}

## Binary search (step 4) starts
Candidate diff: 0.0104167


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0104167, mid=0.0104167, abs_max=50.8192024230957
rel_dist={4: [-47.18011545875409, 47.1801154587541]}

## Binary search (step 5) starts
Candidate diff: 0.0052083


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0052083, mid=0.0052083, abs_max=50.8192024230957
rel_dist={4: [-47.17998488872922, 47.17998488872922]}

## Binary search (step 6) starts
Candidate diff: 0.0026042


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0026042, mid=0.0026042, abs_max=50.8192024230957
rel_dist={4: [-47.17987755544799, 47.17987755544799]}

## Binary search (step 7) starts
Candidate diff: 0.0013021


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0013021, mid=0.0013021, abs_max=50.8192024230957
rel_dist={4: [-47.17981562158482, 47.17981562158482]}

## Binary search (step 8) starts
Candidate diff: 0.0006510


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0006510, mid=0.0006510, abs_max=50.8192024230957
rel_dist={4: [-47.17978462893851, 47.17978462893852]}

## Binary search (step 9) starts
Candidate diff: 0.0003255


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0003255, mid=0.0003255, abs_max=50.8192024230957
rel_dist={4: [-47.1797691207741, 47.17976912077411]}

## Binary search (step 10) starts
Candidate diff: 0.0001628


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0001628, mid=0.0001628, abs_max=50.8192024230957
rel_dist={4: [-47.17976137052537, 47.17976137052537]}

## Binary search (step 11) starts
Candidate diff: 0.0000814


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000814, mid=0.0000814, abs_max=50.8192024230957
rel_dist={4: [-47.1797565820611, 47.1797565820611]}

## Binary search (step 12) starts
Candidate diff: 0.0000407


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000407, mid=0.0000407, abs_max=50.8192024230957
rel_dist={4: [-47.1797538508996, 47.1797538508996]}

## Binary search (step 13) starts
Candidate diff: 0.0000203


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000203, mid=0.0000203, abs_max=50.8192024230957
rel_dist={4: [-47.1797524853568, 47.1797524853568]}

## Binary search (step 14) starts
Candidate diff: 0.0000102


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000102, mid=0.0000102, abs_max=50.8192024230957
rel_dist={4: [-47.179751802659716, 47.17975180265972]}

## Binary search (step 15) starts
Candidate diff: 0.0000051


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000051, mid=0.0000051, abs_max=50.8192024230957
rel_dist={4: [-47.17975146446403, 47.17975146145385]}

## Binary search (step 16) starts
Candidate diff: 0.0000025


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000025, mid=0.0000025, abs_max=50.8192024230957
rel_dist={4: [-47.17975134422682, 47.17975129111434]}

## Binary search (step 17) starts
Candidate diff: 0.0000013


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000013, mid=0.0000013, abs_max=50.8192024230957
rel_dist={4: [-47.17975127539424, 47.17975124487347]}

## Binary search (step 18) starts
Candidate diff: 0.0000006


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000006, mid=0.0000006, abs_max=50.8192024230957
rel_dist={4: [-47.17975118653416, 47.17975122758499]}

## Binary Search Result
Binary search time: 63.95 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1132.47 seconds

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1690313, upper bound: 47.1787782
time: 0.65 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.35 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.35
Output dim: 4, lower bound: -47.1690313, upper bound: 47.1787782
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.35
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -8.8739548, 30.5588493, -4.3289332, 15.6493454, -24.5232983, 34.8877831
1: -10.2695408, 35.3635216, -4.9227071, 18.1375618, -28.4071026, 40.2862282
2: -10.8957434, 34.6628456, -5.4355350, 17.6055393, -28.5012817, 40.0983772
3: -15.7225132, 37.0795212, -7.7107296, 19.0719624, -34.7944641, 44.7902489
4: -17.3079681, 33.5112419, -9.0420084, 16.7446327, -34.0525970, 42.5532455

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.53 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.56 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -8.8639011, 30.5256615, -13.8669882, 47.8264809, -56.5944214, 44.3926430
1: -10.2576370, 35.3249588, -16.4424095, 55.6123199, -65.7206268, 51.7673607
2: -10.8837032, 34.6249809, -17.0157623, 54.3669052, -65.1386719, 51.6407433
3: -15.7045870, 37.0393791, -25.0840302, 58.4735527, -74.0339813, 62.1234093
4: -17.2896118, 33.4746056, -27.2452755, 52.4822922, -69.7719040, 60.7198792

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.74 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.99 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -4.3289332, 15.6493454, -4.3289332, 15.6493454, -19.9782753, 19.9782753
1: -4.9227071, 18.1375618, -4.9227071, 18.1375618, -23.0602684, 23.0602684
2: -5.4355350, 17.6055393, -5.4355350, 17.6055393, -23.0410748, 23.0410748
3: -7.7107296, 19.0719624, -7.7107296, 19.0719624, -26.7826920, 26.7826920
4: -9.0420084, 16.7446327, -9.0420084, 16.7446327, -25.7866364, 25.7866364

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1650497, upper bound: 47.1713854
time: 0.59 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678131, upper bound: 47.1712628
time: 0.51 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -13.8669882, 47.8264809, -4.3289332, 15.6493454, -29.5163288, 52.0300331
1: -16.4424095, 55.6123199, -4.9227071, 18.1375618, -34.5799713, 60.3761864
2: -17.0157623, 54.3669052, -5.4355350, 17.6055393, -34.6212921, 59.6714478
3: -25.0840302, 58.4735527, -7.7107296, 19.0719624, -44.1559906, 66.0495911
4: -27.2452755, 52.4822922, -9.0420084, 16.7446327, -43.9899063, 61.5242958

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0704312, upper bound: 47.1375595
time: 0.72 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0686796, upper bound: 47.1279210
time: 0.51 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -4.3289332, 15.6493454, -13.8669882, 47.8264809, -52.0300293, 29.5163288
1: -4.9227071, 18.1375618, -16.4424095, 55.6123199, -60.3761864, 34.5799713
2: -5.4355350, 17.6055393, -17.0157623, 54.3669052, -59.6714478, 34.6212921
3: -7.7107296, 19.0719624, -25.0840302, 58.4735527, -66.0495911, 44.1559906
4: -9.0420084, 16.7446327, -27.2452755, 52.4822922, -61.5242958, 43.9899063

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1208902, upper bound: 47.0698009
time: 0.81 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.55 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -13.8669882, 47.8264809, -13.8669882, 47.8264809, -61.5810852, 61.5810928
1: -16.4424095, 55.6123199, -16.4424095, 55.6123199, -71.8540726, 71.8540726
2: -17.0157623, 54.3669052, -17.0157623, 54.3669052, -71.2083588, 71.2083588
3: -25.0840302, 58.4735527, -25.0840302, 58.4735527, -83.3268814, 83.3268738
4: -27.2452755, 52.4822922, -27.2452755, 52.4822922, -79.6028137, 79.6028061

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0698009, upper bound: 47.1208902
time: 0.71 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.50 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.95 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -47.1650497, upper bound: 47.1713854
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -47.1678131, upper bound: 47.1712628
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -47.0704312, upper bound: 47.1375595
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -47.0686796, upper bound: 47.1279210
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -47.1208902, upper bound: 47.0698009
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -47.0698009, upper bound: 47.1208902
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -4.0885248, 14.8552752, -4.3289332, 15.6493454, -19.7378674, 19.1842060
1: -4.6401472, 17.2174129, -4.9227071, 18.1375618, -22.7777100, 22.1401196
2: -5.1397719, 16.6970844, -5.4355350, 17.6055393, -22.7453098, 22.1326199
3: -7.2818685, 18.1143341, -7.7107296, 19.0719624, -26.3538303, 25.8250637
4: -8.5971518, 15.8510361, -9.0420084, 16.7446327, -25.3417854, 24.8930435

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1784000, upper bound: 47.1687990
time: 0.48 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1783311, upper bound: 47.1693049
time: 0.53 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -6.2036738, 21.3862705, -4.3081412, 15.5805140, -21.7841873, 25.6944122
1: -7.0725803, 24.8031158, -4.8965139, 18.0587425, -25.1313229, 29.6996307
2: -7.6664505, 24.1312027, -5.4108105, 17.5265007, -25.1929512, 29.5420113
3: -10.9312239, 26.0610123, -7.6718740, 18.9904766, -29.9216995, 33.7328796
4: -12.4161844, 23.1557178, -9.0049934, 16.6665764, -29.0827599, 32.1607094

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_A1_A2_A1

### Relational analysis result of IS_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1673852, upper bound: 47.1679135
time: 0.55 seconds

## Relational analysis of IS_B1_A1_A2_A2

### Relational analysis result of IS_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678911, upper bound: 47.1678911
time: 0.93 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.8669882, 47.8264809, -3.7740495, 13.7595510, -27.6265392, 51.4690704
1: -16.4424095, 55.6123199, -4.2787447, 15.9366035, -32.3790092, 59.7282104
2: -17.0157623, 54.3669052, -4.7472148, 15.4557161, -32.4714737, 58.9725571
3: -25.0840302, 58.4735527, -6.7212100, 16.7551899, -41.8392181, 65.0467453
4: -27.2452755, 52.4822922, -7.9449539, 14.6762562, -41.9215317, 60.4102173

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0704312, upper bound: 47.1375595
time: 0.48 seconds

## Relational analysis of IS_B1_A2_B1_B2

### Relational analysis result of IS_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0698864, upper bound: 47.1270330
time: 0.74 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.5763950, 46.8174782, -3.0988472, 11.2521486, -24.8285427, 49.7928429
1: -16.0870075, 54.4341431, -3.5240169, 12.9181213, -29.0051289, 57.7992554
2: -16.6610355, 53.2107620, -3.8821015, 12.6087656, -29.2698021, 56.9812737
3: -24.5488319, 57.2415276, -5.4794450, 13.5591316, -38.1079636, 62.6080742
4: -26.6886139, 51.3616829, -6.4332042, 12.0126858, -38.7012978, 57.7948875

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0686796, upper bound: 47.1279210
time: 0.53 seconds

## Relational analysis of IS_B1_A2_B2_B2

### Relational analysis result of IS_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0686720, upper bound: 47.1278918
time: 0.55 seconds

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: -3.7740495, 13.7595510, -13.8669882, 47.8264809, -51.4690742, 27.6265392
1: -4.2787447, 15.9366035, -16.4424095, 55.6123199, -59.7282143, 32.3790092
2: -4.7472148, 15.4557161, -17.0157623, 54.3669052, -58.9725571, 32.4714775
3: -6.7212100, 16.7551899, -25.0840302, 58.4735527, -65.0467453, 41.8392181
4: -7.9449539, 14.6762562, -27.2452755, 52.4822922, -60.4102135, 41.9215317

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A1_A1

### Relational analysis result of IS_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1375595, upper bound: 47.0704312
time: 0.76 seconds

## Relational analysis of IS_B2_A1_A1_A2

### Relational analysis result of IS_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1270330, upper bound: 47.0698864
time: 0.55 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -3.0988472, 11.2521486, -13.5763950, 46.8174782, -49.7928429, 24.8285427
1: -3.5240169, 12.9181213, -16.0870075, 54.4341431, -57.7992554, 29.0051289
2: -3.8821015, 12.6087656, -16.6610355, 53.2107620, -56.9812737, 29.2698021
3: -5.4794450, 13.5591316, -24.5488319, 57.2415276, -62.6080742, 38.1079636
4: -6.4332042, 12.0126858, -26.6886139, 51.3616829, -57.7948875, 38.7012978

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A2_A1

### Relational analysis result of IS_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1279210, upper bound: 47.0686796
time: 0.49 seconds

## Relational analysis of IS_B2_A1_A2_A2

### Relational analysis result of IS_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1278918, upper bound: 47.0686720
time: 0.62 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.8669882, 47.8264809, -12.9033422, 44.5502892, -58.2349396, 60.6151619
1: -16.4424095, 55.6123199, -15.2967863, 51.7916565, -67.9424210, 70.7065353
2: -17.0157623, 54.3669052, -15.8526278, 50.6367798, -67.3982315, 70.0347900
3: -25.0840302, 58.4735527, -23.3680229, 54.4814682, -79.2481918, 81.5973206
4: -27.2452755, 52.4822922, -25.4215183, 48.8891449, -75.9544296, 77.7477570

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.88 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.80 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.5763950, 46.8174782, -11.2247467, 38.5842094, -52.1606064, 57.9294586
1: -16.0870075, 54.4341431, -13.2530212, 44.7912140, -60.8487549, 67.5069580
2: -16.6610355, 53.2107620, -13.7886238, 43.8066788, -60.4395447, 66.8377991
3: -24.5488319, 57.2415276, -20.2533207, 47.1470337, -71.6194763, 77.2952347
4: -26.6886139, 51.3616829, -22.0918694, 42.3267174, -68.9953918, 73.3664703

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.87 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.92 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.31 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 4, lower bound: -47.1784000, upper bound: 47.1687990
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 4, lower bound: -47.1783311, upper bound: 47.1693049
IS_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 4, lower bound: -47.1673852, upper bound: 47.1679135
IS_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 4, lower bound: -47.1678911, upper bound: 47.1678911
IS_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 4, lower bound: -47.0704312, upper bound: 47.1375595
IS_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 4, lower bound: -47.0698864, upper bound: 47.1270330
IS_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 4, lower bound: -47.0686796, upper bound: 47.1279210
IS_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 4, lower bound: -47.0686720, upper bound: 47.1278918
IS_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 4, lower bound: -47.1375595, upper bound: 47.0704312
IS_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 4, lower bound: -47.1270330, upper bound: 47.0698864
IS_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 4, lower bound: -47.1279210, upper bound: 47.0686796
IS_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 4, lower bound: -47.1278918, upper bound: 47.0686720
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -3.9784095, 14.5018768, -3.5011101, 12.9874105, -16.9658203, 18.0029812
1: -4.5113440, 16.8087025, -4.0240164, 15.0041294, -19.5154724, 20.8327179
2: -5.0036120, 16.2960930, -4.3955851, 14.6139641, -19.6175766, 20.6916752
3: -7.0868239, 17.6821938, -6.3240385, 15.7447224, -22.8315468, 24.0062332
4: -8.3875227, 15.4578352, -7.3757968, 13.8620195, -22.2495403, 22.8336315

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1784000, upper bound: 47.1674341
time: 0.55 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1784000, upper bound: 47.1687990
time: 0.51 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -4.0885248, 14.8552752, -4.1940603, 15.1722717, -19.2607956, 19.0493355
1: -4.6401472, 17.2174129, -4.7692404, 17.5731640, -22.2133102, 21.9866524
2: -5.1397719, 16.6970844, -5.2681537, 17.0648670, -22.2046375, 21.9652386
3: -7.2818685, 18.1143341, -7.4714885, 18.4803181, -25.7621861, 25.5858231
4: -8.5971518, 15.8510361, -8.7648716, 16.2336521, -24.8308029, 24.6159077

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B1_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1783311, upper bound: 47.1693049
time: 0.74 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1783311, upper bound: 47.1693049
time: 0.53 seconds

## BFS IS instance: IS_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -5.1154051, 17.9620304, -4.2001276, 15.2331467, -20.3485527, 22.1621590
1: -5.8550353, 20.7860451, -4.7700357, 17.6571274, -23.5121593, 25.5560760
2: -6.3262277, 20.2429676, -5.2776732, 17.1318302, -23.4580574, 25.5206413
3: -9.0907803, 21.7931328, -7.4807258, 18.5664139, -27.6571922, 29.2738590
4: -10.2859144, 19.3818359, -8.7992954, 16.2801437, -26.5660591, 28.1811314

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B1_A1_A2_A1_B1

### Relational analysis result of IS_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660203, upper bound: 47.1679135
time: 0.58 seconds

## Relational analysis of IS_B1_A1_A2_A1_B2

### Relational analysis result of IS_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1673852, upper bound: 47.1679135
time: 0.99 seconds

## BFS IS instance: IS_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -6.1029205, 21.0067959, -4.3081412, 15.5805140, -21.6834335, 25.3149376
1: -6.9584203, 24.3565216, -4.8965139, 18.0587425, -25.0171604, 29.2530365
2: -7.5369096, 23.6974564, -5.4108105, 17.5265007, -25.0634098, 29.1082668
3: -10.7420139, 25.5904827, -7.6718740, 18.9904766, -29.7324905, 33.2623520
4: -12.1946831, 22.7528343, -9.0049934, 16.6665764, -28.8612595, 31.7578259

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B1_A1_A2_A2_B1

### Relational analysis result of IS_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678911, upper bound: 47.1678911
time: 0.89 seconds

## Relational analysis of IS_B1_A1_A2_A2_B2

### Relational analysis result of IS_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678911, upper bound: 47.1678911
time: 0.74 seconds

## BFS IS instance: IS_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -13.8035622, 47.6235123, -2.0113435, 7.7563057, -21.5598679, 49.4957848
1: -16.3675251, 55.3771248, -2.2725708, 8.9546232, -25.3221474, 57.4758759
2: -16.9405003, 54.1354790, -2.5653844, 8.6716900, -25.6121902, 56.5561066
3: -24.9736481, 58.2284317, -3.6655130, 9.3957415, -34.3693886, 61.7410851
4: -27.1324806, 52.2557907, -4.4627728, 8.1537590, -35.2862396, 56.6943436

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_B1_A1

### Relational analysis result of IS_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0656443, upper bound: 47.1223860
time: 0.82 seconds

## Relational analysis of IS_B1_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B1_B1_A1

### Relational analysis result of IS_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0651089, upper bound: 47.1348601
time: 0.80 seconds

## Relational analysis of IS_B1_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B1_B1_A1

### Relational analysis result of IS_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0704312, upper bound: 47.1375595
time: 0.50 seconds

## Relational analysis of IS_B1_A2_B1_B1_A2

### Relational analysis result of IS_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0704312, upper bound: 47.1375595
time: 0.48 seconds

## BFS IS instance: IS_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -13.8669882, 47.8264809, -3.4310551, 12.6927109, -26.5597000, 51.1199951
1: -16.4424095, 55.6123199, -3.8694642, 14.7091389, -31.1515446, 59.3104019
2: -17.0157623, 54.3669052, -4.3324442, 14.2426577, -31.2584190, 58.5397606
3: -25.0840302, 58.4735527, -6.1116376, 15.4628716, -40.5469017, 64.4152603
4: -27.2452755, 52.4822922, -7.3231769, 13.4742565, -40.7195320, 59.7215042

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B1_B2_A1

### Relational analysis result of IS_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0698864, upper bound: 47.1270330
time: 0.57 seconds

## Relational analysis of IS_B1_A2_B1_B2_A2

### Relational analysis result of IS_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0698864, upper bound: 47.1270330
time: 0.82 seconds

## BFS IS instance: IS_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -13.5127525, 46.6150322, -1.8940698, 7.2977133, -20.8104649, 48.3783913
1: -16.0122566, 54.1999702, -2.1619205, 8.3386297, -24.3508873, 56.1971550
2: -16.5860157, 52.9802361, -2.3958216, 8.1317167, -24.7177315, 55.2526550
3: -24.4386292, 56.9970665, -3.4469028, 8.6708288, -33.1094589, 60.3165512
4: -26.5761375, 51.1359787, -4.0698967, 7.6648026, -34.2409401, 55.2058754

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B1_A1

### Relational analysis result of IS_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0633252, upper bound: 47.1246086
time: 0.55 seconds

## Relational analysis of IS_B1_A2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B2_B1_B1

### Relational analysis result of IS_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0686796, upper bound: 47.1279210
time: 0.55 seconds

## Relational analysis of IS_B1_A2_B2_B1_B2

### Relational analysis result of IS_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0686478, upper bound: 47.1275148
time: 0.53 seconds

## BFS IS instance: IS_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -13.5763950, 46.8174782, -2.7552955, 10.1911345, -23.7675285, 49.4482574
1: -16.0870075, 54.4341431, -3.1268630, 11.6953678, -27.7823753, 57.4019737
2: -16.6610355, 53.2107620, -3.4718761, 11.4126606, -28.0736961, 56.5607376
3: -24.5488319, 57.2415276, -4.8926821, 12.2728815, -36.8217125, 62.0084648
4: -26.6886139, 51.3616829, -5.8205090, 10.8325176, -37.5211220, 57.1821899

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B2_A1

### Relational analysis result of IS_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0633080, upper bound: 47.1245417
time: 0.50 seconds

## Relational analysis of IS_B1_A2_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B2_B2_B1

### Relational analysis result of IS_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0686720, upper bound: 47.1278918
time: 0.59 seconds

## Relational analysis of IS_B1_A2_B2_B2_B2

### Relational analysis result of IS_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0686392, upper bound: 47.1274168
time: 0.50 seconds

## BFS IS instance: IS_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -2.0113435, 7.7563057, -13.8035622, 47.6235123, -49.4957848, 21.5598679
1: -2.2725708, 8.9546232, -16.3675251, 55.3771248, -57.4758759, 25.3221474
2: -2.5653844, 8.6716900, -16.9405003, 54.1354790, -56.5561066, 25.6121902
3: -3.6655130, 9.3957415, -24.9736481, 58.2284317, -61.7410851, 34.3693886
4: -4.4627728, 8.1537590, -27.1324806, 52.2557907, -56.6943398, 35.2862396

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1223860, upper bound: 47.0656443
time: 0.48 seconds

## Relational analysis of IS_B2_A1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1348601, upper bound: 47.0651089
time: 0.58 seconds

## Relational analysis of IS_B2_A1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1375595, upper bound: 47.0704312
time: 0.79 seconds

## Relational analysis of IS_B2_A1_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1375595, upper bound: 47.0704312
time: 0.77 seconds

## BFS IS instance: IS_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -3.4310551, 12.6927109, -13.8669882, 47.8264809, -51.1199913, 26.5597000
1: -3.8694642, 14.7091389, -16.4424095, 55.6123199, -59.3104057, 31.1515465
2: -4.3324442, 14.2426577, -17.0157623, 54.3669052, -58.5397606, 31.2584190
3: -6.1116376, 15.4628716, -25.0840302, 58.4735527, -64.4152603, 40.5469017
4: -7.3231769, 13.4742565, -27.2452755, 52.4822922, -59.7215042, 40.7195320

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_A1_A2_B1

### Relational analysis result of IS_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1270330, upper bound: 47.0698864
time: 0.88 seconds

## Relational analysis of IS_B2_A1_A1_A2_B2

### Relational analysis result of IS_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1270330, upper bound: 47.0698864
time: 0.55 seconds

## BFS IS instance: IS_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -1.8940698, 7.2977133, -13.5127525, 46.6150322, -48.3783913, 20.8104630
1: -2.1619205, 8.3386297, -16.0122566, 54.1999702, -56.1971550, 24.3508873
2: -2.3958216, 8.1317167, -16.5860157, 52.9802361, -55.2526550, 24.7177315
3: -3.4469028, 8.6708288, -24.4386292, 56.9970665, -60.3165512, 33.1094551
4: -4.0698967, 7.6648026, -26.5761375, 51.1359787, -55.2058754, 34.2409401

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A1_B1

### Relational analysis result of IS_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1246086, upper bound: 47.0633252
time: 0.52 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A2_A1_A1

### Relational analysis result of IS_B2_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1279210, upper bound: 47.0686796
time: 0.45 seconds

## Relational analysis of IS_B2_A1_A2_A1_A2

### Relational analysis result of IS_B2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1275148, upper bound: 47.0686478
time: 0.72 seconds

## BFS IS instance: IS_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -2.7552955, 10.1911345, -13.5763950, 46.8174782, -49.4482613, 23.7675285
1: -3.1268630, 11.6953678, -16.0870075, 54.4341431, -57.4019737, 27.7823753
2: -3.4718761, 11.4126606, -16.6610355, 53.2107620, -56.5607376, 28.0736961
3: -4.8926821, 12.2728815, -24.5488319, 57.2415276, -62.0084648, 36.8217125
4: -5.8205090, 10.8325176, -26.6886139, 51.3616829, -57.1821899, 37.5211220

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A2_B1

### Relational analysis result of IS_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1245417, upper bound: 47.0633080
time: 0.56 seconds

## Relational analysis of IS_B2_A1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A2_A2_A1

### Relational analysis result of IS_B2_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1278918, upper bound: 47.0686720
time: 0.56 seconds

## Relational analysis of IS_B2_A1_A2_A2_A2

### Relational analysis result of IS_B2_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1274168, upper bound: 47.0686392
time: 0.55 seconds

## BFS IS instance: IS_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.9033422, 44.5502892, -12.9033422, 44.5502892, -57.2690163, 57.2690163
1: -15.2967863, 51.7916565, -15.2967863, 51.7916565, -66.7948837, 66.7948837
2: -15.8526278, 50.6367798, -15.8526278, 50.6367798, -66.2246628, 66.2246628
3: -23.3680229, 54.4814682, -23.3680229, 54.4814682, -77.5186310, 77.5186310
4: -25.4215183, 48.8891449, -25.4215183, 48.8891449, -74.0993805, 74.0993805

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B1_A1_B1

### Relational analysis result of IS_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0698009, upper bound: 47.1208902
time: 0.56 seconds

## Relational analysis of IS_B2_A2_B1_A1_B2

### Relational analysis result of IS_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0697881, upper bound: 47.1206535
time: 0.90 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -11.2247467, 38.5842094, -12.9033422, 44.5502892, -55.5802765, 51.4875488
1: -13.2530212, 44.7912140, -15.2967863, 51.7916565, -64.7610168, 60.0561333
2: -13.7886238, 43.8066788, -15.8526278, 50.6367798, -64.1732864, 59.6215973
3: -20.2533207, 47.1470337, -23.3680229, 54.4814682, -74.4376373, 70.4234238
4: -22.0918694, 42.3267174, -25.4215183, 48.8891449, -70.8311157, 67.6940155

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0698009, upper bound: 47.1208902
time: 0.60 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0697881, upper bound: 47.1206535
time: 0.49 seconds

## BFS IS instance: IS_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.9033422, 44.5502892, -11.2247467, 38.5842094, -51.4875488, 55.5802765
1: -15.2967863, 51.7916565, -13.2530212, 44.7912140, -60.0561333, 64.7610092
2: -15.8526278, 50.6367798, -13.7886238, 43.8066788, -59.6216011, 64.1732864
3: -23.3680229, 54.4814682, -20.2533207, 47.1470337, -70.4234238, 74.4376373
4: -25.4215183, 48.8891449, -22.0918694, 42.3267174, -67.6940079, 70.8311157

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0393201, upper bound: 47.0401747
time: 0.81 seconds

## Relational analysis of IS_B2_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.51 seconds

## BFS IS instance: IS_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -11.2247467, 38.5842094, -11.2247467, 38.5842094, -49.8089523, 49.8089523
1: -13.2530212, 44.7912140, -13.2530212, 44.7912140, -58.0222549, 58.0222549
2: -13.7886238, 43.8066788, -13.7886238, 43.8066788, -57.5702248, 57.5702248
3: -20.2533207, 47.1470337, -20.2533207, 47.1470337, -67.3424301, 67.3424301
4: -22.0918694, 42.3267174, -22.0918694, 42.3267174, -64.4185867, 64.4185867

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0393201, upper bound: 47.0401747
time: 0.61 seconds

## Relational analysis of IS_B2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.88 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.11 seconds
IS_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.1784000, upper bound: 47.1674341
IS_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.1784000, upper bound: 47.1687990
IS_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.1783311, upper bound: 47.1693049
IS_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.1783311, upper bound: 47.1693049
IS_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.1660203, upper bound: 47.1679135
IS_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.1673852, upper bound: 47.1679135
IS_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.1678911, upper bound: 47.1678911
IS_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.1678911, upper bound: 47.1678911
IS_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.0704312, upper bound: 47.1375595
IS_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.0704312, upper bound: 47.1375595
IS_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.0698864, upper bound: 47.1270330
IS_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.0698864, upper bound: 47.1270330
IS_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.0686796, upper bound: 47.1279210
IS_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.0686478, upper bound: 47.1275148
IS_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.0686720, upper bound: 47.1278918
IS_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.0686392, upper bound: 47.1274168
IS_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.1375595, upper bound: 47.0704312
IS_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.1375595, upper bound: 47.0704312
IS_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.1270330, upper bound: 47.0698864
IS_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.1270330, upper bound: 47.0698864
IS_B2_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.1279210, upper bound: 47.0686796
IS_B2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.1275148, upper bound: 47.0686478
IS_B2_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.1278918, upper bound: 47.0686720
IS_B2_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.1274168, upper bound: 47.0686392
IS_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.0698009, upper bound: 47.1208902
IS_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.0697881, upper bound: 47.1206535
IS_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.0698009, upper bound: 47.1208902
IS_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.0697881, upper bound: 47.1206535
IS_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.0393201, upper bound: 47.0401747
IS_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
IS_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.0393201, upper bound: 47.0401747
IS_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536

## BFS IS instance: IS_B1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -3.9784095, 14.5018768, -3.3339679, 12.4404860, -16.4188919, 17.8358440
1: -4.5113440, 16.8087025, -3.8323221, 14.3667116, -18.8780537, 20.6410255
2: -5.0036120, 16.2960930, -4.1936383, 13.9953842, -18.9989967, 20.4897308
3: -7.0868239, 17.6821938, -6.0318007, 15.0859957, -22.1728191, 23.7139950
4: -8.3875227, 15.4578352, -7.0685482, 13.2598248, -21.6473465, 22.5263824

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_A1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660427, upper bound: 47.1660596
time: 0.52 seconds

## Relational analysis of IS_B1_A1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660427, upper bound: 47.1674341
time: 0.51 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -3.9784095, 14.5018768, -5.0702901, 17.8143559, -21.7927628, 19.5721645
1: -4.5113440, 16.8087025, -5.8005986, 20.6206131, -25.1319580, 22.6093006
2: -5.0036120, 16.2960930, -6.2775421, 20.0659771, -25.0695877, 22.5736332
3: -7.0868239, 17.6821938, -9.0183678, 21.6345825, -28.7214069, 26.7005596
4: -8.3875227, 15.4578352, -10.2168274, 19.2128162, -27.6003380, 25.6746635

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A1_B1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1767279, upper bound: 47.1656485
time: 0.54 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1731355, upper bound: 47.1652099
time: 0.69 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -4.0885248, 14.8552752, -3.9750445, 14.4460850, -18.5346088, 18.8303204
1: -4.6401472, 17.2174129, -4.5114536, 16.7320309, -21.3721771, 21.7288666
2: -5.1397719, 16.6970844, -4.9959445, 16.2338371, -21.3736038, 21.6930294
3: -7.2818685, 18.1143341, -7.0781264, 17.6019325, -24.8838005, 25.1924610
4: -8.5971518, 15.8510361, -8.3550758, 15.4140797, -24.0112305, 24.2061119

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A1_B2_B1_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1565703, upper bound: 47.1610615
time: 0.76 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_A1_A1_B2_B1_A1

### Relational analysis result of IS_B1_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660203, upper bound: 47.1679304
time: 0.81 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_A2

### Relational analysis result of IS_B1_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660203, upper bound: 47.1693049
time: 0.76 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -4.0885248, 14.8552752, -6.1029205, 21.0067959, -25.0953197, 20.9581947
1: -4.6401472, 17.2174129, -6.9584203, 24.3565216, -28.9966698, 24.1758327
2: -5.1397719, 16.6970844, -7.5369096, 23.6974564, -28.8372288, 24.2339935
3: -7.2818685, 18.1143341, -10.7420139, 25.5904827, -32.8723488, 28.8563480
4: -8.5971518, 15.8510361, -12.1946831, 22.7528343, -31.3499870, 28.0457191

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A1_B2_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1766474, upper bound: 47.1660148
time: 0.86 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1731142, upper bound: 47.1655762
time: 0.48 seconds

## BFS IS instance: IS_B1_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -5.1154051, 17.9620304, -3.9773979, 14.4985437, -19.6139469, 21.9394283
1: -5.8550353, 20.7860451, -4.5101190, 16.8048763, -22.6599064, 25.2961636
2: -6.3262277, 20.2429676, -5.0023842, 16.2922897, -22.6185169, 25.2453499
3: -9.0907803, 21.7931328, -7.0849619, 17.6781998, -26.7689800, 28.8780899
4: -10.2859144, 19.3818359, -8.3856688, 15.4540949, -25.7400093, 27.7675037

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_A1_A2_A1_B1_B1

### Relational analysis result of IS_B1_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660203, upper bound: 47.1660427
time: 0.55 seconds

## Relational analysis of IS_B1_A1_A2_A1_B1_B2

### Relational analysis result of IS_B1_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1673852, upper bound: 47.1679135
time: 0.88 seconds

## BFS IS instance: IS_B1_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -5.1154051, 17.9620304, -6.0863447, 21.0169506, -26.1323547, 24.0483742
1: -5.8550353, 20.7860451, -6.9371142, 24.3732815, -30.2283134, 27.7231579
2: -6.3262277, 20.2429676, -7.5245743, 23.7097321, -30.0359592, 27.7675419
3: -9.0907803, 21.7931328, -10.7284508, 25.6067982, -34.6975784, 32.5215836
4: -10.2859144, 19.3818359, -12.1950521, 22.7442551, -33.0301704, 31.5768890

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_A1_A2_A1_B2_B1

### Relational analysis result of IS_B1_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1673852, upper bound: 47.1673696
time: 0.56 seconds

## Relational analysis of IS_B1_A1_A2_A1_B2_B2

### Relational analysis result of IS_B1_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1673852, upper bound: 47.1679135
time: 0.62 seconds

## BFS IS instance: IS_B1_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -6.1029205, 21.0067959, -4.0885248, 14.8552752, -20.9581947, 25.0953197
1: -6.9584203, 24.3565216, -4.6401472, 17.2174129, -24.1758327, 28.9966698
2: -7.5369096, 23.6974564, -5.1397719, 16.6970844, -24.2339916, 28.8372288
3: -10.7420139, 25.5904827, -7.2818685, 18.1143341, -28.8563480, 32.8723488
4: -12.1946831, 22.7528343, -8.5971518, 15.8510361, -28.0457191, 31.3499870

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A1_A2_A2_B1_B1

### Relational analysis result of IS_B1_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1662038, upper bound: 47.1614269
time: 0.48 seconds

## Relational analysis of IS_B1_A1_A2_A2_B1_B2

### Relational analysis result of IS_B1_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1656462, upper bound: 47.1656461
time: 0.81 seconds

## BFS IS instance: IS_B1_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -6.1029205, 21.0067959, -6.2036738, 21.3862705, -27.4891911, 27.2104702
1: -6.9584203, 24.3565216, -7.0725803, 24.8031158, -31.7615356, 31.4291000
2: -7.5369096, 23.6974564, -7.6664505, 24.1312027, -31.6681118, 31.3639069
3: -10.7420139, 25.5904827, -10.9312239, 26.0610123, -36.8030167, 36.5217056
4: -12.1946831, 22.7528343, -12.4161844, 23.1557178, -35.3504028, 35.1690140

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_A1_A2_A2_B2_B1

### Relational analysis result of IS_B1_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678911, upper bound: 47.1672928
time: 0.83 seconds

## Relational analysis of IS_B1_A1_A2_A2_B2_B2

### Relational analysis result of IS_B1_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678911, upper bound: 47.1678543
time: 0.55 seconds

## BFS IS instance: IS_B1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -12.8399019, 44.3466225, -2.0113435, 7.7563057, -20.5962067, 46.1489029
1: -15.2216196, 51.5554466, -2.2725708, 8.9546232, -24.1762428, 53.5631523
2: -15.7771111, 50.4045219, -2.5653844, 8.6716900, -24.4488010, 52.7451172
3: -23.2569027, 54.2353973, -3.6655130, 9.3957415, -32.6526451, 57.6613617
4: -25.3082886, 48.6620140, -4.4627728, 8.1537590, -33.4620476, 53.0451431

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B1_B1_A1_A1

### Relational analysis result of IS_B1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0651089, upper bound: 47.1348601
time: 0.78 seconds

## Relational analysis of IS_B1_A2_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_B1_A1_A1

### Relational analysis result of IS_B1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0656443, upper bound: 47.1223860
time: 0.55 seconds

## Relational analysis of IS_B1_A2_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B1_B1_A1_A1

### Relational analysis result of IS_B1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0701561, upper bound: 47.1364444
time: 0.58 seconds

## Relational analysis of IS_B1_A2_B1_B1_A1_A2

### Relational analysis result of IS_B1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0704312, upper bound: 47.1375563
time: 0.84 seconds

## BFS IS instance: IS_B1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -11.1694784, 38.4077187, -2.0113435, 7.7563057, -18.9257832, 40.4190636
1: -13.1874046, 44.5872459, -2.2725708, 8.9546232, -22.1420269, 46.8571968
2: -13.7230539, 43.6055717, -2.5653844, 8.6716900, -22.3947449, 46.1709557
3: -20.1565781, 46.9337349, -3.6655130, 9.3957415, -29.5523186, 50.5992432
4: -21.9935703, 42.1293983, -4.4627728, 8.1537590, -30.1473293, 46.5921707

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B1_B1_A2_A1

### Relational analysis result of IS_B1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0651089, upper bound: 47.1348601
time: 0.82 seconds

## Relational analysis of IS_B1_A2_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_B1_A2_A1

### Relational analysis result of IS_B1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0656443, upper bound: 47.1223860
time: 0.77 seconds

## Relational analysis of IS_B1_A2_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B1_B1_A2_A1

### Relational analysis result of IS_B1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0701561, upper bound: 47.1364444
time: 0.78 seconds

## Relational analysis of IS_B1_A2_B1_B1_A2_A2

### Relational analysis result of IS_B1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0704312, upper bound: 47.1375563
time: 0.58 seconds

## BFS IS instance: IS_B1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -12.9033422, 44.5502892, -3.4310551, 12.6927109, -25.5960541, 47.7738419
1: -15.2967863, 51.7916565, -3.8694642, 14.7091389, -30.0059223, 55.3987617
2: -15.8526278, 50.6367798, -4.3324442, 14.2426577, -30.0952854, 54.7296295
3: -23.3680229, 54.4814682, -6.1116376, 15.4628716, -38.8308792, 60.3365593
4: -25.4215183, 48.8891449, -7.3231769, 13.4742565, -38.8957634, 56.0731277

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B1_B2_A1_B1

### Relational analysis result of IS_B1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0689556, upper bound: 47.1270330
time: 0.51 seconds

## Relational analysis of IS_B1_A2_B1_B2_A1_B2

### Relational analysis result of IS_B1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0698864, upper bound: 47.1270330
time: 0.52 seconds

## BFS IS instance: IS_B1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -11.2247467, 38.5842094, -3.4310551, 12.6927109, -23.9174576, 42.0152626
1: -13.2530212, 44.7912140, -3.8694642, 14.7091389, -27.9621601, 48.6600113
2: -13.7886238, 43.8066788, -4.3324442, 14.2426577, -28.0312805, 48.1265678
3: -20.2533207, 47.1470337, -6.1116376, 15.4628716, -35.7161827, 53.2413559
4: -22.0918694, 42.3267174, -7.3231769, 13.4742565, -35.5661240, 49.6498947

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B1_B2_A2_B1

### Relational analysis result of IS_B1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0689556, upper bound: 47.1270330
time: 0.84 seconds

## Relational analysis of IS_B1_A2_B1_B2_A2_B2

### Relational analysis result of IS_B1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0698864, upper bound: 47.1270330
time: 0.51 seconds

## BFS IS instance: IS_B1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -13.5127525, 46.6150322, -1.3845830, 5.5557604, -19.0685101, 47.8671494
1: -16.0122566, 54.1999702, -1.5911571, 6.3327122, -22.3449688, 55.6242218
2: -16.5860157, 52.9802361, -1.7506844, 6.1586976, -22.7447128, 54.6063957
3: -24.4386292, 56.9970665, -2.5644333, 6.5570183, -30.9956436, 59.4371262
4: -26.5761375, 51.1359787, -3.0642345, 5.7635617, -32.3396988, 54.2002144

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B2_B1_B1_A1

### Relational analysis result of IS_B1_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0683689, upper bound: 47.1263813
time: 0.49 seconds

## Relational analysis of IS_B1_A2_B2_B1_B1_A2

### Relational analysis result of IS_B1_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0683689, upper bound: 47.1275148
time: 0.50 seconds

## BFS IS instance: IS_B1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -13.5127525, 46.6150322, -1.8537294, 7.1764112, -20.6891613, 48.3394356
1: -16.0122566, 54.1999702, -2.1186795, 8.1974497, -24.2097034, 56.1540565
2: -16.5860157, 52.9802361, -2.3488302, 7.9920363, -24.5780525, 55.2056389
3: -24.4386292, 56.9970665, -3.3832459, 8.5229225, -32.9615479, 60.2496376
4: -26.5761375, 51.1359787, -3.9991758, 7.5297847, -34.1059227, 55.1351547

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_B1_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B1_B2_A1

### Relational analysis result of IS_B1_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0632846, upper bound: 47.1241568
time: 0.60 seconds

## Relational analysis of IS_B1_A2_B2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B2_B1_B2_A1

### Relational analysis result of IS_B1_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0686478, upper bound: 47.1275148
time: 0.57 seconds

## Relational analysis of IS_B1_A2_B2_B1_B2_A2

### Relational analysis result of IS_B1_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0686478, upper bound: 47.1275148
time: 0.57 seconds

## BFS IS instance: IS_B1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -13.5763950, 46.8174782, -2.2056880, 8.4149694, -21.9913635, 48.8926315
1: -16.0870075, 54.4341431, -2.5051420, 9.6535006, -25.7405071, 56.7781258
2: -16.6610355, 53.2107620, -2.8067064, 9.4117737, -26.0728054, 55.8940277
3: -24.5488319, 57.2415276, -3.9490108, 10.1583872, -34.7072182, 61.0647545
4: -26.6886139, 51.3616829, -4.8294826, 8.8776760, -35.5662918, 56.1911659

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_B1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B2_B1_A1

### Relational analysis result of IS_B1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0633080, upper bound: 47.1245417
time: 0.53 seconds

## Relational analysis of IS_B1_A2_B2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_B2_B2_B1_A1

### Relational analysis result of IS_B1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0664906, upper bound: 47.1213108
time: 0.55 seconds

## Relational analysis of IS_B1_A2_B2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B2_B2_B1_A1

### Relational analysis result of IS_B1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0683603, upper bound: 47.1262833
time: 0.49 seconds

## Relational analysis of IS_B1_A2_B2_B2_B1_A2

### Relational analysis result of IS_B1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0683603, upper bound: 47.1274168
time: 0.52 seconds

## BFS IS instance: IS_B1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -13.5763950, 46.8174782, -2.6747246, 9.9570904, -23.5334835, 49.3680191
1: -16.0870075, 54.4341431, -3.0293925, 11.4253492, -27.5123558, 57.3046417
2: -16.6610355, 53.2107620, -3.3799801, 11.1443720, -27.8054085, 56.4680290
3: -24.5488319, 57.2415276, -4.7521524, 11.9871187, -36.5359497, 61.8645477
4: -26.6886139, 51.3616829, -5.6850152, 10.5674877, -37.2560959, 57.0466957

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B2_B2_A1

### Relational analysis result of IS_B1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0632682, upper bound: 47.1240494
time: 0.50 seconds

## Relational analysis of IS_B1_A2_B2_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_B1_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_B1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B2_B2_B2_A1

### Relational analysis result of IS_B1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0683603, upper bound: 47.1262833
time: 0.54 seconds

## Relational analysis of IS_B1_A2_B2_B2_B2_A2

### Relational analysis result of IS_B1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0683603, upper bound: 47.1274168
time: 0.65 seconds

## BFS IS instance: IS_B2_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -2.0113435, 7.7563057, -12.8399019, 44.3466225, -46.1489029, 20.5962067
1: -2.2725708, 8.9546232, -15.2216196, 51.5554466, -53.5631523, 24.1762428
2: -2.5653844, 8.6716900, -15.7771111, 50.4045219, -52.7451172, 24.4488010
3: -3.6655130, 9.3957415, -23.2569027, 54.2353973, -57.6613617, 32.6526451
4: -4.4627728, 8.1537590, -25.3082886, 48.6620140, -53.0451469, 33.4620476

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A1_A1_B1_B1

### Relational analysis result of IS_B2_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1348601, upper bound: 47.0651089
time: 0.52 seconds

## Relational analysis of IS_B2_A1_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_A1_A1_B1_B1

### Relational analysis result of IS_B2_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1223860, upper bound: 47.0656443
time: 0.78 seconds

## Relational analysis of IS_B2_A1_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_B2_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A1_A1_B1_B1

### Relational analysis result of IS_B2_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1364444, upper bound: 47.0701561
time: 0.83 seconds

## Relational analysis of IS_B2_A1_A1_A1_B1_B2

### Relational analysis result of IS_B2_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1375563, upper bound: 47.0704312
time: 0.54 seconds

## BFS IS instance: IS_B2_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -2.0113435, 7.7563057, -11.1694784, 38.4077187, -40.4190636, 18.9257832
1: -2.2725708, 8.9546232, -13.1874046, 44.5872459, -46.8571968, 22.1420269
2: -2.5653844, 8.6716900, -13.7230539, 43.6055717, -46.1709557, 22.3947449
3: -3.6655130, 9.3957415, -20.1565781, 46.9337349, -50.5992432, 29.5523186
4: -4.4627728, 8.1537590, -21.9935703, 42.1293983, -46.5921707, 30.1473293

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A1_A1_B2_B1

### Relational analysis result of IS_B2_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1348601, upper bound: 47.0651089
time: 0.56 seconds

## Relational analysis of IS_B2_A1_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_B2_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_A1_A1_B2_B1

### Relational analysis result of IS_B2_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1223860, upper bound: 47.0656443
time: 0.45 seconds

## Relational analysis of IS_B2_A1_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A1_A1_B2_B1

### Relational analysis result of IS_B2_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1364444, upper bound: 47.0701561
time: 0.88 seconds

## Relational analysis of IS_B2_A1_A1_A1_B2_B2

### Relational analysis result of IS_B2_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1375563, upper bound: 47.0704312
time: 0.59 seconds

## BFS IS instance: IS_B2_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -3.4310551, 12.6927109, -12.9033422, 44.5502892, -47.7738419, 25.5960541
1: -3.8694642, 14.7091389, -15.2967863, 51.7916565, -55.3987617, 30.0059223
2: -4.3324442, 14.2426577, -15.8526278, 50.6367798, -54.7296295, 30.0952854
3: -6.1116376, 15.4628716, -23.3680229, 54.4814682, -60.3365593, 38.8308792
4: -7.3231769, 13.4742565, -25.4215183, 48.8891449, -56.0731277, 38.8957634

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A1_A2_B1_A1

### Relational analysis result of IS_B2_A1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1270330, upper bound: 47.0689556
time: 0.93 seconds

## Relational analysis of IS_B2_A1_A1_A2_B1_A2

### Relational analysis result of IS_B2_A1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1270330, upper bound: 47.0698864
time: 0.54 seconds

## BFS IS instance: IS_B2_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -3.4310551, 12.6927109, -11.2247467, 38.5842094, -42.0152626, 23.9174576
1: -3.8694642, 14.7091389, -13.2530212, 44.7912140, -48.6600113, 27.9621582
2: -4.3324442, 14.2426577, -13.7886238, 43.8066788, -48.1265678, 28.0312805
3: -6.1116376, 15.4628716, -20.2533207, 47.1470337, -53.2413559, 35.7161789
4: -7.3231769, 13.4742565, -22.0918694, 42.3267174, -49.6498947, 35.5661240

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A1_A2_B2_A1

### Relational analysis result of IS_B2_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1270330, upper bound: 47.0689556
time: 0.55 seconds

## Relational analysis of IS_B2_A1_A1_A2_B2_A2

### Relational analysis result of IS_B2_A1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1270330, upper bound: 47.0698864
time: 0.52 seconds

## BFS IS instance: IS_B2_A1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -1.3845830, 5.5557604, -13.5127525, 46.6150322, -47.8671494, 19.0685081
1: -1.5911571, 6.3327122, -16.0122566, 54.1999702, -55.6242218, 22.3449688
2: -1.7506844, 6.1586976, -16.5860157, 52.9802361, -54.6063957, 22.7447128
3: -2.5644333, 6.5570183, -24.4386292, 56.9970665, -59.4371262, 30.9956436
4: -3.0642345, 5.7635617, -26.5761375, 51.1359787, -54.2002144, 32.3396988

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A2_A1_A1_B1

### Relational analysis result of IS_B2_A1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1263813, upper bound: 47.0683689
time: 0.56 seconds

## Relational analysis of IS_B2_A1_A2_A1_A1_B2

### Relational analysis result of IS_B2_A1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1263813, upper bound: 47.0686478
time: 0.89 seconds

## BFS IS instance: IS_B2_A1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -1.8537294, 7.1764112, -13.5127525, 46.6150322, -48.3394356, 20.6891632
1: -2.1186795, 8.1974497, -16.0122566, 54.1999702, -56.1540565, 24.2097054
2: -2.3488302, 7.9920363, -16.5860157, 52.9802361, -55.2056389, 24.5780525
3: -3.3832459, 8.5229225, -24.4386292, 56.9970665, -60.2496376, 32.9615517
4: -3.9991758, 7.5297847, -26.5761375, 51.1359787, -55.1351547, 34.1059227

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_B2_A1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A1_A2_B1

### Relational analysis result of IS_B2_A1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1241568, upper bound: 47.0632846
time: 0.55 seconds

## Relational analysis of IS_B2_A1_A2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_A2_A1_A2_B1

### Relational analysis result of IS_B2_A1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1275148, upper bound: 47.0686478
time: 0.76 seconds

## Relational analysis of IS_B2_A1_A2_A1_A2_B2

### Relational analysis result of IS_B2_A1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1275148, upper bound: 47.0686478
time: 0.83 seconds

## BFS IS instance: IS_B2_A1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -2.2056880, 8.4149694, -13.5763950, 46.8174782, -48.8926353, 21.9913635
1: -2.5051420, 9.6535006, -16.0870075, 54.4341431, -56.7781258, 25.7405052
2: -2.8067064, 9.4117737, -16.6610355, 53.2107620, -55.8940277, 26.0728054
3: -3.9490108, 10.1583872, -24.5488319, 57.2415276, -61.0647545, 34.7072182
4: -4.8294826, 8.8776760, -26.6886139, 51.3616829, -56.1911621, 35.5662918

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_B2_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A2_A1_B1

### Relational analysis result of IS_B2_A1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1245417, upper bound: 47.0633080
time: 0.58 seconds

## Relational analysis of IS_B2_A1_A2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_A2_A2_A1_B1

### Relational analysis result of IS_B2_A1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1213108, upper bound: 47.0664906
time: 0.52 seconds

## Relational analysis of IS_B2_A1_A2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_B2_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A2_A2_A1_B1

### Relational analysis result of IS_B2_A1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1262833, upper bound: 47.0683603
time: 0.89 seconds

## Relational analysis of IS_B2_A1_A2_A2_A1_B2

### Relational analysis result of IS_B2_A1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1262833, upper bound: 47.0686392
time: 0.64 seconds

## BFS IS instance: IS_B2_A1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -2.6747246, 9.9570904, -13.5763950, 46.8174782, -49.3680153, 23.5334835
1: -3.0293925, 11.4253492, -16.0870075, 54.4341431, -57.3046417, 27.5123558
2: -3.3799801, 11.1443720, -16.6610355, 53.2107620, -56.4680252, 27.8054085
3: -4.7521524, 11.9871187, -24.5488319, 57.2415276, -61.8645477, 36.5359497
4: -5.6850152, 10.5674877, -26.6886139, 51.3616829, -57.0466995, 37.2560959

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A2_A2_B1

### Relational analysis result of IS_B2_A1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1240494, upper bound: 47.0632682
time: 0.79 seconds

## Relational analysis of IS_B2_A1_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_B2_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_B2_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A2_A2_A2_B1

### Relational analysis result of IS_B2_A1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1262833, upper bound: 47.0683603
time: 0.49 seconds

## Relational analysis of IS_B2_A1_A2_A2_A2_B2

### Relational analysis result of IS_B2_A1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1262833, upper bound: 47.0686392
time: 0.57 seconds

## BFS IS instance: IS_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -12.8399019, 44.3466225, -8.5782623, 30.1781197, -42.7906914, 52.7066574
1: -15.2216196, 51.5554466, -10.0983152, 35.1235161, -50.0151405, 61.3498688
2: -15.7771111, 50.4045219, -10.6518869, 34.2437057, -49.7197876, 60.7712593
3: -23.2569027, 54.2353973, -15.6297207, 37.0361481, -59.9293442, 69.5236969
4: -25.3082886, 48.6620140, -17.4083576, 32.9491119, -58.0285149, 65.8380508

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1499879, upper bound: 47.1499879
time: 0.58 seconds

## Relational analysis of IS_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1499879, upper bound: 47.1499879
time: 0.57 seconds

## BFS IS instance: IS_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.9033422, 44.5502892, -12.4929514, 43.2831039, -55.8670654, 56.8638229
1: -15.2967863, 51.7916565, -14.8166914, 50.3266754, -65.1651001, 66.3101196
2: -15.8526278, 50.6367798, -15.3689508, 49.1942673, -64.6338120, 65.7295532
3: -23.3680229, 54.4814682, -22.6567554, 52.9362526, -75.8129959, 76.7890244
4: -25.4215183, 48.8891449, -24.6819305, 47.4811783, -72.5894089, 73.2986832

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1236745, upper bound: 47.1439686
time: 0.62 seconds

## Relational analysis of IS_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1463077, upper bound: 47.1463077
time: 0.55 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -11.1694784, 38.4077187, -8.5782623, 30.1781197, -41.1112099, 46.9780960
1: -13.1874046, 44.5872459, -10.0983152, 35.1235161, -47.9909935, 54.6439133
2: -13.7230539, 43.6055717, -10.6518869, 34.2437057, -47.6785965, 54.1997414
3: -20.1565781, 46.9337349, -15.6297207, 37.0361481, -56.8627701, 62.4619522
4: -21.9935703, 42.1293983, -17.4083576, 32.9491119, -54.7751884, 59.4629555

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0695241, upper bound: 47.1197830
time: 0.52 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0698009, upper bound: 47.1208902
time: 0.57 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -11.2247467, 38.5842094, -12.4929514, 43.2831039, -54.1783295, 51.0771599
1: -13.2530212, 44.7912140, -14.8166914, 50.3266754, -63.1312141, 59.5713806
2: -13.7886238, 43.8066788, -15.3689508, 49.1942673, -62.5824432, 59.1264763
3: -20.2533207, 47.1470337, -22.6567554, 52.9362526, -72.7320023, 69.6938171
4: -22.0918694, 42.3267174, -24.6819305, 47.4811783, -69.3211441, 66.8933258

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0695092, upper bound: 47.1195199
time: 0.58 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0697881, upper bound: 47.1206535
time: 0.82 seconds

## BFS IS instance: IS_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -12.9033422, 44.5502892, -10.7382069, 37.0713997, -49.9747391, 55.0912094
1: -15.2967863, 51.7916565, -12.6549892, 43.0521164, -58.3090591, 64.1587372
2: -15.8526278, 50.6367798, -13.2225256, 42.0720711, -57.8821907, 63.6062698
3: -23.3680229, 54.4814682, -19.3875694, 45.3159103, -68.5843353, 73.5707626
4: -25.4215183, 48.8891449, -21.2321262, 40.6393242, -66.0032959, 69.9638596

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1197830, upper bound: 47.0695241
time: 0.52 seconds

## Relational analysis of IS_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1195199, upper bound: 47.0695092
time: 0.76 seconds

## BFS IS instance: IS_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.9033422, 44.5502892, -10.9528446, 37.6870155, -50.5903511, 55.3092041
1: -15.2967863, 51.7916565, -12.9132843, 43.7553520, -59.0161285, 64.4216614
2: -15.8526278, 50.6367798, -13.4681273, 42.7748718, -58.5841064, 63.8511238
3: -23.3680229, 54.4814682, -19.7512875, 46.0691261, -69.3389740, 73.9344635
4: -25.4215183, 48.8891449, -21.6068287, 41.3166008, -66.6798401, 70.3380051

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1208902, upper bound: 47.0698009
time: 0.54 seconds

## Relational analysis of IS_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1206535, upper bound: 47.0697881
time: 0.57 seconds

## BFS IS instance: IS_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -11.2247467, 38.5842094, -10.7382069, 37.0713997, -48.2961426, 49.3224182
1: -13.2530212, 44.7912140, -12.6549892, 43.0521164, -56.2751770, 57.4199791
2: -13.7886238, 43.8066788, -13.2225256, 42.0720711, -55.8308220, 57.0032082
3: -20.2533207, 47.1470337, -19.3875694, 45.3159103, -65.5033340, 66.4755554
4: -22.0918694, 42.3267174, -21.2321262, 40.6393242, -62.7311935, 63.5584984

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0390412
time: 0.52 seconds

## Relational analysis of IS_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0401747
time: 0.86 seconds

## BFS IS instance: IS_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -11.2247467, 38.5842094, -10.9528446, 37.6870155, -48.9117546, 49.5370522
1: -13.2530212, 44.7912140, -12.9132843, 43.7553520, -56.9822502, 57.6828995
2: -13.7886238, 43.8066788, -13.4681273, 42.7748718, -56.5327339, 57.2480659
3: -20.2533207, 47.1470337, -19.7512875, 46.0691261, -66.2579727, 66.8392715
4: -22.0918694, 42.3267174, -21.6068287, 41.3166008, -63.4084702, 63.9326515

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0401747, upper bound: 47.0393201
time: 0.52 seconds

## Relational analysis of IS_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0401747, upper bound: 47.0404536
time: 0.57 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.27 seconds
IS_B1_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1660427, upper bound: 47.1660596
IS_B1_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1660427, upper bound: 47.1674341
IS_B1_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1767279, upper bound: 47.1656485
IS_B1_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1731355, upper bound: 47.1652099
IS_B1_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1660203, upper bound: 47.1679304
IS_B1_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1660203, upper bound: 47.1693049
IS_B1_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1766474, upper bound: 47.1660148
IS_B1_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1731142, upper bound: 47.1655762
IS_B1_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1660203, upper bound: 47.1660427
IS_B1_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1673852, upper bound: 47.1679135
IS_B1_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1673852, upper bound: 47.1673696
IS_B1_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1673852, upper bound: 47.1679135
IS_B1_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1662038, upper bound: 47.1614269
IS_B1_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1656462, upper bound: 47.1656461
IS_B1_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1678911, upper bound: 47.1672928
IS_B1_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1678911, upper bound: 47.1678543
IS_B1_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0701561, upper bound: 47.1364444
IS_B1_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0704312, upper bound: 47.1375563
IS_B1_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0701561, upper bound: 47.1364444
IS_B1_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0704312, upper bound: 47.1375563
IS_B1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0689556, upper bound: 47.1270330
IS_B1_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0698864, upper bound: 47.1270330
IS_B1_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0689556, upper bound: 47.1270330
IS_B1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0698864, upper bound: 47.1270330
IS_B1_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0683689, upper bound: 47.1263813
IS_B1_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0683689, upper bound: 47.1275148
IS_B1_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0686478, upper bound: 47.1275148
IS_B1_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0686478, upper bound: 47.1275148
IS_B1_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0683603, upper bound: 47.1262833
IS_B1_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0683603, upper bound: 47.1274168
IS_B1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0683603, upper bound: 47.1262833
IS_B1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0683603, upper bound: 47.1274168
IS_B2_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1364444, upper bound: 47.0701561
IS_B2_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1375563, upper bound: 47.0704312
IS_B2_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1364444, upper bound: 47.0701561
IS_B2_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1375563, upper bound: 47.0704312
IS_B2_A1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1270330, upper bound: 47.0689556
IS_B2_A1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1270330, upper bound: 47.0698864
IS_B2_A1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1270330, upper bound: 47.0689556
IS_B2_A1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1270330, upper bound: 47.0698864
IS_B2_A1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1263813, upper bound: 47.0683689
IS_B2_A1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1263813, upper bound: 47.0686478
IS_B2_A1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1275148, upper bound: 47.0686478
IS_B2_A1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1275148, upper bound: 47.0686478
IS_B2_A1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1262833, upper bound: 47.0683603
IS_B2_A1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1262833, upper bound: 47.0686392
IS_B2_A1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1262833, upper bound: 47.0683603
IS_B2_A1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1262833, upper bound: 47.0686392
IS_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1499879, upper bound: 47.1499879
IS_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1499879, upper bound: 47.1499879
IS_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1236745, upper bound: 47.1439686
IS_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1463077, upper bound: 47.1463077
IS_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0695241, upper bound: 47.1197830
IS_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0698009, upper bound: 47.1208902
IS_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0695092, upper bound: 47.1195199
IS_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0697881, upper bound: 47.1206535
IS_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1197830, upper bound: 47.0695241
IS_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1195199, upper bound: 47.0695092
IS_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1208902, upper bound: 47.0698009
IS_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.1206535, upper bound: 47.0697881
IS_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0390412
IS_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0401747
IS_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0401747, upper bound: 47.0393201
IS_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.27
Output dim: 4, lower bound: -47.0401747, upper bound: 47.0404536

## BFS IS instance: IS_B1_A1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -3.3339679, 12.4404860, -3.3339679, 12.4404860, -15.7744541, 15.7744541
1: -3.8323221, 14.3667116, -3.8323221, 14.3667116, -18.1990299, 18.1990318
2: -4.1936383, 13.9953842, -4.1936383, 13.9953842, -18.1890221, 18.1890202
3: -6.0318007, 15.0859957, -6.0318007, 15.0859957, -21.1177959, 21.1177959
4: -7.0685482, 13.2598248, -7.0685482, 13.2598248, -20.3283730, 20.3283730

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42

Time for candidate selection: 8.32 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 22

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 13

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 13

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 8
type: A, layer: 5, pos: 8
type: A, layer: 5, pos: 31
type: B, layer: 5, pos: 31
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 48
type: B, layer: 5, pos: 48
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 42
type: A, layer: 5, pos: 42
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 45
type: A, layer: 5, pos: 45
type: A, layer: 5, pos: 0
type: B, layer: 5, pos: 0
type: A, layer: 5, pos: 36
type: B, layer: 5, pos: 36
type: B, layer: 5, pos: 39
type: A, layer: 5, pos: 39
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 32
type: B, layer: 5, pos: 32
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 26
type: B, layer: 5, pos: 22
type: A, layer: 5, pos: 22
type: A, layer: 5, pos: 41
type: B, layer: 5, pos: 41
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 25

Time for candidate selection: 23.87 seconds

### Candidate
type: B, layer: 5, pos: 8

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 8

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 31

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 31

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 46

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1601367, upper bound: 47.1623598
time: 0.56 seconds

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_A2

### Relational analysis result of IS_B1_A1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1623596, upper bound: 47.1623596
time: 0.59 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -3.9750445, 14.4460850, -3.3339679, 12.4404860, -16.4155293, 17.7800522
1: -4.5114536, 16.7320309, -3.8323221, 14.3667116, -18.8781605, 20.5643520
2: -4.9959445, 16.2338371, -4.1936383, 13.9953842, -18.9913292, 20.4274731
3: -7.0781264, 17.6019325, -6.0318007, 15.0859957, -22.1641216, 23.6337299
4: -8.3550758, 15.4140797, -7.0685482, 13.2598248, -21.6149006, 22.4826279

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 18

Time for candidate selection: 8.70 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 22

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 13

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 13

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 18

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 31
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 31
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 48
type: B, layer: 5, pos: 48
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 42
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 8
type: B, layer: 5, pos: 42
type: A, layer: 5, pos: 0
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 45
type: A, layer: 5, pos: 36
type: B, layer: 5, pos: 0
type: B, layer: 5, pos: 45
type: B, layer: 5, pos: 36
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 39
type: B, layer: 5, pos: 39
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 26
type: B, layer: 5, pos: 32
type: A, layer: 5, pos: 32
type: A, layer: 5, pos: 22
type: B, layer: 5, pos: 41
type: A, layer: 5, pos: 41
type: B, layer: 5, pos: 22
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 25

Time for candidate selection: 25.49 seconds

### Candidate
type: A, layer: 5, pos: 31

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 46

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1

### Relational analysis result of IS_B1_A1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1601367, upper bound: 47.1637343
time: 0.86 seconds

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A2

### Relational analysis result of IS_B1_A1_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1623596, upper bound: 47.1637343
time: 0.56 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -2.2503781, 8.9455576, -5.0565610, 17.7666035, -20.0169811, 14.0021172
1: -2.5083449, 10.3775654, -5.7844944, 20.5648766, -23.0732174, 16.1620598
2: -2.8935421, 9.9744968, -6.2603841, 20.0116806, -22.9052219, 16.2348804
3: -4.0368142, 10.9438505, -8.9932861, 21.5762367, -25.6130505, 19.9371376
4: -5.1629324, 9.2298565, -10.1898108, 19.1602135, -24.3231449, 19.4196663

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_A1_A1_B1_B2_A1_A1

### Relational analysis result of IS_B1_A1_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1717526, upper bound: 47.1638358
time: 0.56 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_A1_A2

### Relational analysis result of IS_B1_A1_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1716589, upper bound: 47.1633929
time: 0.81 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -3.8320017, 14.0057850, -5.0702901, 17.8143559, -21.6463585, 19.0760746
1: -4.3430710, 16.2366734, -5.8005986, 20.6206131, -24.9636841, 22.0372677
2: -4.8248506, 15.7351351, -6.2775421, 20.0659771, -24.8908272, 22.0126724
3: -6.8286271, 17.0918865, -9.0183678, 21.6345825, -28.4632092, 26.1102524
4: -8.1160364, 14.9158669, -10.2168274, 19.2128162, -27.3288479, 25.1326923

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_A1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_A1_B1_B2_A2_A1

### Relational analysis result of IS_B1_A1_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1731176, upper bound: 47.1646968
time: 0.49 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_A2_A2

### Relational analysis result of IS_B1_A1_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1731176, upper bound: 47.1652099
time: 0.81 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -3.3339679, 12.4404860, -3.9750445, 14.4460850, -17.7800522, 16.4155312
1: -3.8323221, 14.3667116, -4.5114536, 16.7320309, -20.5643520, 18.8781605
2: -4.1936383, 13.9953842, -4.9959445, 16.2338371, -20.4274731, 18.9913292
3: -6.0318007, 15.0859957, -7.0781264, 17.6019325, -23.6337299, 22.1641216
4: -7.0685482, 13.2598248, -8.3550758, 15.4140797, -22.4826279, 21.6149006

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 18

Time for candidate selection: 8.72 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 22

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1666667, mid=0.1666667, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 1) starts
Candidate diff: 0.0833333


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1752727, upper bound: 47.1684808
time: 0.49 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679581, upper bound: 47.1679581
time: 0.49 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.15 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.15
Output dim: 4, lower bound: -47.1752727, upper bound: 47.1684808
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.15
Output dim: 4, lower bound: -47.1679581, upper bound: 47.1679581

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.3289332, 15.6493454, -7.0806756, 24.7960377, -29.1249695, 22.7300167
1: -4.9227071, 18.1375618, -8.1733112, 28.7153568, -33.6380653, 26.3108730
2: -5.4355350, 17.6055393, -8.7818174, 28.0788403, -33.5143738, 26.3873558
3: -7.7107296, 19.0719624, -12.6078606, 30.1267490, -37.8374786, 31.6798229
4: -9.0420084, 16.7446327, -14.1291494, 27.0256519, -36.0676460, 30.8737793

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679581, upper bound: 47.1679581
time: 0.50 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679581, upper bound: 47.1679581
time: 0.50 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -13.8669882, 47.8264809, -8.6101542, 29.6928444, -43.5598297, 56.3413658
1: -16.4424095, 55.6123199, -9.9591999, 34.3566132, -50.7990112, 65.4222336
2: -17.0157623, 54.3669052, -10.5797205, 33.6746674, -50.6904259, 64.8336792
3: -25.0840302, 58.4735527, -15.2551098, 36.0304794, -61.1145096, 73.5841751
4: -27.2452755, 52.4822922, -16.8258591, 32.5553856, -59.8006592, 69.3081512

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0642450, upper bound: 47.0834364
time: 0.45 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.15 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 4, lower bound: -47.1679581, upper bound: 47.1679581
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 4, lower bound: -47.1679581, upper bound: 47.1679581
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 4, lower bound: -47.0642450, upper bound: 47.0834364
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -4.3289332, 15.6493454, -4.3289332, 15.6493454, -19.9782753, 19.9782753
1: -4.9227071, 18.1375618, -4.9227071, 18.1375618, -23.0602684, 23.0602684
2: -5.4355350, 17.6055393, -5.4355350, 17.6055393, -23.0410748, 23.0410748
3: -7.7107296, 19.0719624, -7.7107296, 19.0719624, -26.7826920, 26.7826920
4: -9.0420084, 16.7446327, -9.0420084, 16.7446327, -25.7866364, 25.7866364

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1740553, upper bound: 47.1684409
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712628, upper bound: 47.1678131
time: 0.86 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -4.3289332, 15.6493454, -12.8257484, 44.2895927, -48.4904671, 28.4750900
1: -4.9227071, 18.1375618, -15.1263590, 51.5547981, -56.3153458, 33.2639198
2: -5.4355350, 17.6055393, -15.7967310, 50.3415146, -55.6429100, 33.4022713
3: -7.7107296, 19.0719624, -23.2090626, 54.2801094, -61.8550987, 42.2810173
4: -9.0420084, 16.7446327, -25.4259567, 48.5490341, -57.5910339, 42.1705894

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1011442, upper bound: 47.0665081
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0880072, upper bound: 47.0638408
time: 0.90 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -13.8669882, 47.8264809, -7.8863921, 27.1971207, -41.0641098, 55.6141243
1: -16.4424095, 55.6123199, -9.1125174, 31.4263744, -47.8687706, 64.5749359
2: -17.0157623, 54.3669052, -9.6920519, 30.8347950, -47.8505478, 63.9380798
3: -25.0840302, 58.4735527, -13.9547367, 32.9564857, -58.0405159, 72.2707214
4: -27.2452755, 52.4822922, -15.4038181, 29.8292179, -57.0744934, 67.8699036

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.52 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -12.7908230, 44.0865097, -6.3875051, 22.0018177, -34.7926369, 50.4084167
1: -15.1312590, 51.2482185, -7.3248172, 25.2959003, -40.4271584, 58.4714050
2: -15.7052488, 50.0906487, -7.8386145, 24.8943634, -40.5996132, 57.8553543
3: -23.1098843, 53.9105415, -11.1716309, 26.4630661, -49.5729523, 64.9940872
4: -25.1840820, 48.3460426, -12.3583727, 24.1137104, -49.2977905, 60.7044106

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0393201, upper bound: 47.0401747
time: 0.52 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.57 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.53 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 4, lower bound: -47.1740553, upper bound: 47.1684409
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 4, lower bound: -47.1712628, upper bound: 47.1678131
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 4, lower bound: -47.1011442, upper bound: 47.0665081
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 4, lower bound: -47.0880072, upper bound: 47.0638408
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 4, lower bound: -47.0393201, upper bound: 47.0401747
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.0885248, 14.8552752, -4.2938104, 15.5331678, -19.6216927, 19.1490841
1: -4.6401472, 17.2174129, -4.8813195, 18.0029831, -22.6431313, 22.0987320
2: -5.1397719, 16.6970844, -5.3923540, 17.4725246, -22.6122952, 22.0894375
3: -7.2818685, 18.1143341, -7.6480060, 18.9319248, -26.2137928, 25.7623386
4: -8.5971518, 15.8510361, -8.9770498, 16.6137981, -25.2109489, 24.8280830

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712896, upper bound: 47.1712896
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712896, upper bound: 47.1712896
time: 0.48 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.2036738, 21.3862705, -4.2358456, 15.3407001, -21.5443745, 25.6221123
1: -7.0725803, 24.8031158, -4.8052192, 17.7845078, -24.8570881, 29.6083317
2: -7.6664505, 24.1312027, -5.3247833, 17.2514839, -24.9179344, 29.4559860
3: -10.9312239, 26.0610123, -7.5364375, 18.7069206, -29.6381454, 33.5974503
4: -12.4161844, 23.1557178, -8.8762884, 16.3948631, -28.8110466, 32.0320053

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1673845, upper bound: 47.1679130
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678906, upper bound: 47.1678906
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.7740495, 13.7595510, -12.8257484, 44.2895927, -47.9295158, 26.5853004
1: -4.2787447, 15.9366035, -15.1263590, 51.5547981, -55.6673698, 31.0629616
2: -4.7472148, 15.4557161, -15.7967310, 50.3415146, -54.9440155, 31.2524471
3: -6.7212100, 16.7551899, -23.2090626, 54.2801094, -60.8522491, 39.9642525
4: -7.9449539, 14.6762562, -25.4259567, 48.5490341, -56.4781723, 40.1022110

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0880072, upper bound: 47.0638408
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0880072, upper bound: 47.0638408
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.0988472, 11.2521486, -12.1088943, 41.7376480, -44.7440643, 23.3610420
1: -3.5240169, 12.9181213, -14.2659130, 48.5555878, -51.9624329, 27.1840343
2: -3.8821015, 12.6087656, -14.9070950, 47.4135971, -51.2166481, 27.5158596
3: -5.4794450, 13.5591316, -21.8750057, 51.1307869, -56.5354385, 35.4341354
4: -6.4332042, 12.0126858, -23.9868622, 45.7307510, -52.1639557, 35.9995422

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0831182, upper bound: 47.0578411
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0878106, upper bound: 47.0632801
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0880072, upper bound: 47.0638408
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.9033422, 44.5502892, -7.8863921, 27.1971207, -40.1004639, 52.2679787
1: -15.2967863, 51.7916565, -9.1125174, 31.4263744, -46.7231560, 60.6632805
2: -15.8526278, 50.6367798, -9.6920519, 30.8347950, -46.6874161, 60.1279449
3: -23.3680229, 54.4814682, -13.9547367, 32.9564857, -56.3245010, 68.1920319
4: -25.4215183, 48.8891449, -15.4038181, 29.8292179, -55.2507248, 64.2215195

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0605090, upper bound: 47.0815670
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0642450, upper bound: 47.0834364
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -11.2247467, 38.5842094, -7.8863921, 27.1971207, -38.4218674, 46.4706001
1: -13.2530212, 44.7912140, -9.1125174, 31.4263744, -44.6793900, 53.9037323
2: -13.7886238, 43.8066788, -9.6920519, 30.8347950, -44.6234169, 53.4987259
3: -20.2533207, 47.1470337, -13.9547367, 32.9564857, -53.2098007, 61.0968323
4: -22.0918694, 42.3267174, -15.4038181, 29.8292179, -51.9210854, 57.7305374

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0605090, upper bound: 47.0815670
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0642450, upper bound: 47.0834364
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -12.7272720, 43.8778877, -5.8594179, 20.1327782, -32.8600502, 49.6717453
1: -15.0537205, 51.0067062, -6.6864014, 23.1041451, -38.1578674, 57.5906525
2: -15.6307783, 49.8517952, -7.2021346, 22.7496414, -38.3804169, 56.9806747
3: -22.9956894, 53.6593513, -10.1838274, 24.2136650, -47.2093430, 63.7499275
4: -25.0718670, 48.1126709, -11.3504562, 22.0592155, -47.1310806, 59.4631271

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0393201, upper bound: 47.0401747
time: 0.78 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0393201, upper bound: 47.0401747
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -12.7908230, 44.0865097, -6.2778120, 21.6416531, -34.4324760, 50.2990837
1: -15.1312590, 51.2482185, -7.1870689, 24.8781242, -40.0093842, 58.3348160
2: -15.7052488, 50.0906487, -7.7067957, 24.4813576, -40.1866074, 57.7228470
3: -23.1098843, 53.9105415, -10.9660168, 26.0270386, -49.1369209, 64.7864456
4: -25.1840820, 48.3460426, -12.1560593, 23.7112980, -48.8953781, 60.5021019

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0401747, upper bound: 47.0393201
time: 0.53 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0401747, upper bound: 47.0404536
time: 0.50 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.67 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 4, lower bound: -47.1712896, upper bound: 47.1712896
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 4, lower bound: -47.1712896, upper bound: 47.1712896
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 4, lower bound: -47.1673845, upper bound: 47.1679130
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 4, lower bound: -47.1678906, upper bound: 47.1678906
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 4, lower bound: -47.0880072, upper bound: 47.0638408
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 4, lower bound: -47.0880072, upper bound: 47.0638408
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 4, lower bound: -47.0878106, upper bound: 47.0632801
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 4, lower bound: -47.0880072, upper bound: 47.0638408
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 4, lower bound: -47.0605090, upper bound: 47.0815670
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 4, lower bound: -47.0642450, upper bound: 47.0834364
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 4, lower bound: -47.0605090, upper bound: 47.0815670
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 4, lower bound: -47.0642450, upper bound: 47.0834364
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 4, lower bound: -47.0393201, upper bound: 47.0401747
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 4, lower bound: -47.0393201, upper bound: 47.0401747
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 4, lower bound: -47.0401747, upper bound: 47.0393201
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.67
Output dim: 4, lower bound: -47.0401747, upper bound: 47.0404536

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.0885248, 14.8552752, -4.0885248, 14.8552752, -18.9437981, 18.9437981
1: -4.6401472, 17.2174129, -4.6401472, 17.2174129, -21.8575592, 21.8575592
2: -5.1397719, 16.6970844, -5.1397719, 16.6970844, -21.8368549, 21.8368549
3: -7.2818685, 18.1143341, -7.2818685, 18.1143341, -25.3962021, 25.3962021
4: -8.5971518, 15.8510361, -8.5971518, 15.8510361, -24.4481888, 24.4481888

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660203, upper bound: 47.1679296
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1747487, upper bound: 47.1689886
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.0885248, 14.8552752, -6.2036738, 21.3862705, -25.4747925, 21.0589485
1: -4.6401472, 17.2174129, -7.0725803, 24.8031158, -29.4432640, 24.2899933
2: -5.1397719, 16.6970844, -7.6664505, 24.1312027, -29.2709732, 24.3635349
3: -7.2818685, 18.1143341, -10.9312239, 26.0610123, -33.3428764, 29.0455589
4: -8.5971518, 15.8510361, -12.4161844, 23.1557178, -31.7528687, 28.2672195

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1748885, upper bound: 47.1684523
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1747487, upper bound: 47.1689886
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -5.1154051, 17.9620304, -3.8520682, 14.1158628, -19.2312679, 21.8140984
1: -5.8550353, 20.7860451, -4.3576918, 16.3672295, -22.2222633, 25.1437378
2: -6.3262277, 20.2429676, -4.8487148, 15.8627377, -22.1889648, 25.0916786
3: -9.0907803, 21.7931328, -6.8588405, 17.2049160, -26.2956944, 28.6519699
4: -10.2859144, 19.3818359, -8.1415586, 15.0315666, -25.3174820, 27.5233955

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1673845, upper bound: 47.1679130
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1673845, upper bound: 47.1679130
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -6.1029205, 21.0067959, -4.2325616, 15.3291349, -21.4320564, 25.2393570
1: -6.9584203, 24.3565216, -4.8014755, 17.7708302, -24.7292461, 29.1579971
2: -7.5369096, 23.6974564, -5.3207216, 17.2383785, -24.7752876, 29.0181770
3: -10.7420139, 25.5904827, -7.5306187, 18.6925945, -29.4346085, 33.1211014
4: -12.1946831, 22.7528343, -8.8695917, 16.3824539, -28.5771351, 31.6224251

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678906, upper bound: 47.1678906
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678906, upper bound: 47.1678906
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.7740495, 13.7595510, -12.1196041, 41.8921280, -45.4619713, 25.8791542
1: -4.2787447, 15.9366035, -14.3085260, 48.7408333, -52.7624626, 30.2451286
2: -4.7472148, 15.4557161, -14.9407234, 47.6119881, -52.1348076, 30.3964386
3: -6.7212100, 16.7551899, -21.9557590, 51.3325081, -57.8167763, 38.7109451
4: -7.9449539, 14.6762562, -24.0498276, 45.9409370, -53.8130341, 38.7260818

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1010531, upper bound: 47.0658690
time: 0.49 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1011442, upper bound: 47.0665081
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.7740495, 13.7595510, -10.9450626, 37.6084023, -41.3824501, 24.7046127
1: -4.2787447, 15.9366035, -12.8934517, 43.6690178, -47.9477615, 28.8300514
2: -4.7472148, 15.4557161, -13.4669304, 42.6933708, -47.4405861, 28.9226456
3: -6.7212100, 16.7551899, -19.7413235, 45.9990692, -52.7202644, 36.4965096
4: -7.9449539, 14.6762562, -21.6055470, 41.2416763, -49.1866302, 36.2818031

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1010531, upper bound: 47.0658690
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1011442, upper bound: 47.0665081
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -2.4840326, 9.2791185, -12.0609760, 41.5800476, -43.9647102, 21.3400955
1: -2.8140178, 10.6496401, -14.2074041, 48.3732109, -51.0658875, 24.8570442
2: -3.1398907, 10.3862915, -14.8501558, 47.2324715, -50.2901001, 25.2364464
3: -4.4031024, 11.2005587, -21.7876968, 50.9399757, -55.2659912, 32.9882545
4: -5.3245440, 9.8329000, -23.9003773, 45.5537033, -50.8782463, 33.7332764

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0829271, upper bound: 47.0570009
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0820300, upper bound: 47.0604392
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0871541, upper bound: 47.0631217
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0871541, upper bound: 47.0632801
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -3.0158601, 11.0064144, -12.1088943, 41.7376480, -44.6614456, 23.1153088
1: -3.4244437, 12.6325045, -14.2659130, 48.5555878, -51.8627319, 26.8984184
2: -3.7862558, 12.3272820, -14.9070950, 47.4135971, -51.1198921, 27.2343712
3: -5.3343902, 13.2565231, -21.8750057, 51.1307869, -56.3868294, 35.1315308
4: -6.2886052, 11.7358017, -23.9868622, 45.7307510, -52.0193558, 35.7226639

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0829271, upper bound: 47.0578411
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0873462, upper bound: 47.0636670
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0873462, upper bound: 47.0638408
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -12.8409367, 44.3440094, -7.3645077, 25.4187469, -38.2596779, 51.5367126
1: -15.2199888, 51.5518188, -8.4769840, 29.3495617, -44.5695457, 59.7844162
2: -15.7789707, 50.3993759, -9.0696068, 28.7901382, -44.5691071, 59.2679787
3: -23.2548542, 54.2326889, -12.9885550, 30.8053379, -54.0601883, 66.9708252
4: -25.3104591, 48.6574974, -14.4219141, 27.8682060, -53.1786652, 63.0000687

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0963546, upper bound: 47.1028247
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1364445, upper bound: 47.1087050
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.9033422, 44.5502892, -7.7413502, 26.7201595, -39.6234970, 52.1230431
1: -15.2967863, 51.7916565, -8.9257545, 30.8739986, -46.1707840, 60.4774551
2: -15.8526278, 50.6367798, -9.5190830, 30.2876854, -46.1403084, 59.9540939
3: -23.3680229, 54.4814682, -13.6782970, 32.3794136, -55.7474289, 67.9133987
4: -25.4215183, 48.8891449, -15.1418896, 29.2957630, -54.7172699, 63.9537163

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0963546, upper bound: 47.1552590
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1364445, upper bound: 47.1550510
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -11.1731358, 38.4118004, -7.3645077, 25.4187469, -36.5918694, 45.7763062
1: -13.1901455, 44.5913925, -8.4769840, 29.3495617, -42.5396957, 53.0683746
2: -13.7279339, 43.6090431, -9.0696068, 28.7901382, -42.5180740, 52.6786499
3: -20.1603642, 46.9391937, -12.9885550, 30.8053379, -50.9656906, 59.9181519
4: -21.9991932, 42.1352234, -14.4219141, 27.8682060, -49.8674011, 56.5571365

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0602298, upper bound: 47.0804335
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0602298, upper bound: 47.0815670
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -11.2247467, 38.5842094, -7.7413502, 26.7201595, -37.9448967, 46.3255577
1: -13.2530212, 44.7912140, -8.9257545, 30.8739986, -44.1270218, 53.7169647
2: -13.7886238, 43.8066788, -9.5190830, 30.2876854, -44.0763092, 53.3257523
3: -20.2533207, 47.1470337, -13.6782970, 32.3794136, -52.6327248, 60.8181763
4: -22.0918694, 42.3267174, -15.1418896, 29.2957630, -51.3876228, 57.4686012

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0639848, upper bound: 47.0823702
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0639848, upper bound: 47.0834364
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -12.8409367, 44.3440094, -5.8594179, 20.1327782, -32.9737167, 50.0249367
1: -15.2199888, 51.5518188, -6.6864014, 23.1041451, -38.3241348, 57.9918060
2: -15.7789707, 50.3993759, -7.2021346, 22.7496414, -38.5286064, 57.4060402
3: -23.2548542, 54.2326889, -10.1838274, 24.2136650, -47.4685211, 64.1907272
4: -25.3104591, 48.6574974, -11.3504562, 22.0592155, -47.3696747, 59.9959679

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0390412
time: 0.92 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0401747
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -11.1731358, 38.4118004, -5.8594179, 20.1327782, -31.3059139, 44.2712173
1: -13.1901455, 44.5913925, -6.6864014, 23.1041451, -36.2942886, 51.2777939
2: -13.7279339, 43.6090431, -7.2021346, 22.7496414, -36.4775696, 50.8111725
3: -20.1603642, 46.9391937, -10.1838274, 24.2136650, -44.3740196, 57.1230202
4: -21.9991932, 42.1352234, -11.3504562, 22.0592155, -44.0584106, 53.4856796

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0390412
time: 0.55 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0401747
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -12.3708954, 42.7764626, -6.2778120, 21.6416531, -34.0125465, 48.9785347
1: -14.6049519, 49.7493706, -7.1870689, 24.8781242, -39.4830780, 56.8255196
2: -15.2114305, 48.5816193, -7.7067957, 24.4813576, -39.6927834, 56.2038918
3: -22.3470840, 52.3244667, -10.9660168, 26.0270386, -48.3741226, 63.1912003
4: -24.4395599, 46.8688126, -12.1560593, 23.7112980, -48.1508560, 59.0248718

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0393201
time: 0.53 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0393201
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -12.5142918, 43.1857300, -6.2778120, 21.6416531, -34.1559448, 49.3911667
1: -14.7906322, 50.2079773, -7.1870689, 24.8781242, -39.6687546, 57.2892036
2: -15.3801060, 49.0575294, -7.7067957, 24.4813576, -39.8614616, 56.6830368
3: -22.6088428, 52.8257599, -10.9660168, 26.0270386, -48.6358795, 63.6941185
4: -24.6927204, 47.3377686, -12.1560593, 23.7112980, -48.4040146, 59.4938278

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0404536
time: 0.79 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0404536
time: 0.84 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.37 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.1660203, upper bound: 47.1679296
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.1747487, upper bound: 47.1689886
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.1748885, upper bound: 47.1684523
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.1747487, upper bound: 47.1689886
IS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.1673845, upper bound: 47.1679130
IS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.1673845, upper bound: 47.1679130
IS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.1678906, upper bound: 47.1678906
IS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.1678906, upper bound: 47.1678906
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.1010531, upper bound: 47.0658690
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.1011442, upper bound: 47.0665081
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.1010531, upper bound: 47.0658690
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.1011442, upper bound: 47.0665081
IS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.0871541, upper bound: 47.0631217
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.0871541, upper bound: 47.0632801
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.0873462, upper bound: 47.0636670
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.0873462, upper bound: 47.0638408
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.0963546, upper bound: 47.1028247
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.1364445, upper bound: 47.1087050
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.0963546, upper bound: 47.1552590
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.1364445, upper bound: 47.1550510
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.0602298, upper bound: 47.0804335
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.0602298, upper bound: 47.0815670
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.0639848, upper bound: 47.0823702
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.0639848, upper bound: 47.0834364
IS_A2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0390412
IS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0401747
IS_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0390412
IS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0401747
IS_A2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0393201
IS_A2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0393201
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0404536
IS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0404536

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.3339679, 12.4404860, -3.6958864, 13.5991192, -16.9330864, 16.1363697
1: -3.8323221, 14.3667116, -4.1818523, 15.7646065, -19.5969276, 18.5485592
2: -4.1936383, 13.9953842, -4.6533270, 15.2726135, -19.4662514, 18.6487122
3: -6.0318007, 15.0859957, -6.5878377, 16.5744305, -22.6062298, 21.6738338
4: -7.0685482, 13.2598248, -7.8455558, 14.4526558, -21.5212040, 21.1053810

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660596, upper bound: 47.1660596
time: 0.48 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660596, upper bound: 47.1749842
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -3.9750445, 14.4460850, -4.0854430, 14.8443670, -18.8194084, 18.5315285
1: -4.5114536, 16.7320309, -4.6366658, 17.2045021, -21.7159538, 21.3686943
2: -4.9959445, 16.2338371, -5.1359477, 16.6847420, -21.6806870, 21.3697834
3: -7.0781264, 17.6019325, -7.2764263, 18.1007996, -25.1789265, 24.8783588
4: -8.3550758, 15.4140797, -8.5908041, 15.8393917, -24.1944675, 24.0048828

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1749842, upper bound: 47.1670978
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1749842, upper bound: 47.1797914
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -3.6958864, 13.5991192, -5.1154051, 17.9620304, -21.6579170, 18.7145233
1: -4.1818523, 15.7646065, -5.8550353, 20.7860451, -24.9678955, 21.6196384
2: -4.6533270, 15.2726135, -6.3262277, 20.2429676, -24.8962936, 21.5988407
3: -6.5878377, 16.5744305, -9.0907803, 21.7931328, -28.3809681, 25.6652107
4: -7.8455558, 14.4526558, -10.2859144, 19.3818359, -27.2273922, 24.7385712

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1701485, upper bound: 47.1652180
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1722538, upper bound: 47.1652034
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -4.0854430, 14.8443670, -6.1029205, 21.0067959, -25.0922375, 20.9472847
1: -4.6366658, 17.2045021, -6.9584203, 24.3565216, -28.9931870, 24.1629219
2: -5.1359477, 16.6847420, -7.5369096, 23.6974564, -28.8334045, 24.2216511
3: -7.2764263, 18.1007996, -10.7420139, 25.5904827, -32.8669090, 28.8428135
4: -8.5908041, 15.8393917, -12.1946831, 22.7528343, -31.3436394, 28.0340748

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1701479, upper bound: 47.1656178
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1720863, upper bound: 47.1655717
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -5.1154051, 17.9620304, -3.6958864, 13.5991192, -18.7145233, 21.6579170
1: -5.8550353, 20.7860451, -4.1818523, 15.7646065, -21.6196384, 24.9678974
2: -6.3262277, 20.2429676, -4.6533270, 15.2726135, -21.5988407, 24.8962936
3: -9.0907803, 21.7931328, -6.5878377, 16.5744305, -25.6652107, 28.3809681
4: -10.2859144, 19.3818359, -7.8455558, 14.4526558, -24.7385712, 27.2273922

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1673845, upper bound: 47.1660427
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1673845, upper bound: 47.1679130
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -5.1154051, 17.9620304, -5.7188907, 19.8396225, -24.9550285, 23.6809216
1: -5.8550353, 20.7860451, -6.5142136, 23.0074196, -28.8624496, 27.3002567
2: -6.3262277, 20.2429676, -7.0898938, 22.3624611, -28.6886883, 27.3328590
3: -9.0907803, 21.7931328, -10.1030264, 24.1830254, -33.2738037, 31.8961544
4: -10.2859144, 19.3818359, -11.5131865, 21.4431820, -31.7290955, 30.8950233

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1673845, upper bound: 47.1673682
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1673845, upper bound: 47.1679130
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -6.1029205, 21.0067959, -4.0718036, 14.7992687, -20.9021893, 25.0785980
1: -6.9584203, 24.3565216, -4.6201611, 17.1527405, -24.1111584, 28.9766827
2: -7.5369096, 23.6974564, -5.1194825, 16.6332359, -24.1701450, 28.8169384
3: -10.7420139, 25.5904827, -7.2513862, 18.0468369, -28.7888508, 32.8418694
4: -12.1946831, 22.7528343, -8.5658922, 15.7888613, -27.9835415, 31.3187256

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678906, upper bound: 47.1660203
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678906, upper bound: 47.1678539
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -6.1029205, 21.0067959, -6.2013359, 21.3773708, -27.4802914, 27.2081299
1: -6.9584203, 24.3565216, -7.0699358, 24.7925529, -31.7509727, 31.4264545
2: -7.5369096, 23.6974564, -7.6633868, 24.1211376, -31.6580467, 31.3608437
3: -10.7420139, 25.5904827, -10.9268322, 26.0499554, -36.7919693, 36.5173111
4: -12.1946831, 22.7528343, -12.4110060, 23.1462994, -35.3409805, 35.1638298

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678906, upper bound: 47.1672916
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678906, upper bound: 47.1678539
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.0475309, 11.4952679, -12.0716343, 41.7329788, -44.5659637, 23.5669022
1: -3.4280396, 13.3304625, -14.2495079, 48.5566635, -51.7200699, 27.5799694
2: -3.8826449, 12.8824883, -14.8835163, 47.4286194, -51.0827179, 27.7660027
3: -5.4488935, 14.0269852, -21.8678856, 51.1395607, -56.3481789, 35.8948593
4: -6.6468649, 12.1436949, -23.9629211, 45.7616920, -52.3265419, 36.1066093

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1529227, upper bound: 47.1569773
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658529, upper bound: 47.1600415
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -3.6656094, 13.4249163, -12.1196041, 41.8921280, -45.3540764, 25.5445213
1: -4.1421041, 15.5496635, -14.3085260, 48.7408333, -52.6254044, 29.8581886
2: -4.6192713, 15.0691986, -14.9407234, 47.6119881, -52.0068169, 30.0099220
3: -6.5204768, 16.3473377, -21.9557590, 51.3325081, -57.6125603, 38.3030930
4: -7.7538829, 14.2926893, -24.0498276, 45.9409370, -53.6144867, 38.3425179

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1529227, upper bound: 47.1579637
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1664410, upper bound: 47.1609763
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.0475309, 11.4952679, -10.8991804, 37.4556351, -40.5031662, 22.3944435
1: -3.4280396, 13.3304625, -12.8379517, 43.4919548, -46.9199944, 26.1684151
2: -3.8826449, 12.8824883, -13.4127293, 42.5182610, -46.4009056, 26.2952137
3: -5.4488935, 14.0269852, -19.6589069, 45.8143959, -51.2632904, 33.6858902
4: -6.6468649, 12.1436949, -21.5224628, 41.0720673, -47.7189331, 33.6661568

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0964517, upper bound: 47.0600962
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1001542, upper bound: 47.0656287
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1001542, upper bound: 47.0658690
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.6656094, 13.4249163, -10.9450626, 37.6084023, -41.2740097, 24.3699780
1: -4.1421041, 15.5496635, -12.8934517, 43.6690178, -47.8111229, 28.4431095
2: -4.6192713, 15.0691986, -13.4669304, 42.6933708, -47.3126411, 28.5361290
3: -6.5204768, 16.3473377, -19.7413235, 45.9990692, -52.5195389, 36.0886574
4: -7.7538829, 14.2926893, -21.6055470, 41.2416763, -48.9955559, 35.8982353

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1002428, upper bound: 47.0662547
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1002428, upper bound: 47.0665081
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -2.4840326, 9.2791185, -11.7050200, 40.5114479, -42.8871078, 20.9841385
1: -2.8140178, 10.6496401, -13.7693815, 47.1506958, -49.8340683, 24.4190197
2: -3.1398907, 10.3862915, -14.4371405, 46.0050049, -49.0533028, 24.8234329
3: -4.4031024, 11.2005587, -21.1499710, 49.6388359, -53.9548645, 32.3505287
4: -5.3245440, 9.8329000, -23.2687454, 44.3553162, -49.6798592, 33.1016464

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0821886, upper bound: 47.0569708
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0808745, upper bound: 47.0586673
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0871541, upper bound: 47.0631217
time: 0.49 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0871541, upper bound: 47.0631217
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -2.4840326, 9.2791185, -11.8990927, 41.0575790, -43.4355125, 21.1782093
1: -2.8140178, 10.6496401, -14.0079870, 47.7680321, -50.4558678, 24.6576252
2: -3.1398907, 10.3862915, -14.6592493, 46.6319160, -49.6836090, 25.0455399
3: -4.4031024, 11.2005587, -21.4907284, 50.3078423, -54.6263771, 32.6912880
4: -5.3245440, 9.8329000, -23.6094570, 44.9688301, -50.2933731, 33.4423561

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0821886, upper bound: 47.0570009
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0808745, upper bound: 47.0586673
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0871541, upper bound: 47.0632801
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0871541, upper bound: 47.0632801
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -3.0158601, 11.0064144, -11.7050200, 40.5114479, -43.4243279, 22.7114334
1: -3.4244437, 12.6325045, -13.7693815, 47.1506958, -50.4469757, 26.4018860
2: -3.7862558, 12.3272820, -14.4371405, 46.0050049, -49.7008629, 26.7644176
3: -5.3343902, 13.2565231, -21.1499710, 49.6388359, -54.8837395, 34.4064941
4: -6.2886052, 11.7358017, -23.2687454, 44.3553162, -50.6439209, 35.0045471

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0823752, upper bound: 47.0578056
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0873462, upper bound: 47.0636670
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0873462, upper bound: 47.0636670
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -3.0158601, 11.0064144, -11.8990927, 41.0575790, -43.9736061, 22.9055061
1: -3.4244437, 12.6325045, -14.0079870, 47.7680321, -51.0690575, 26.6404915
2: -3.7862558, 12.3272820, -14.6592493, 46.6319160, -50.3311729, 26.9865284
3: -5.3343902, 13.2565231, -21.4907284, 50.3078423, -55.5552521, 34.7472534
4: -6.2886052, 11.7358017, -23.6094570, 44.9688301, -51.2574348, 35.3452568

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0823752, upper bound: 47.0578411
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0873462, upper bound: 47.0637900
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0873462, upper bound: 47.0637900
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.5198402, 40.0604782, -6.9647646, 24.0557518, -35.5755920, 46.8348236
1: -13.6847801, 46.5408363, -7.9971671, 27.7489471, -41.4337273, 54.2770653
2: -14.1587362, 45.5369911, -8.5767965, 27.2410145, -41.3997498, 53.8949852
3: -20.9280624, 48.9484749, -12.2533998, 29.1213474, -50.0494080, 60.9218521
4: -22.7320061, 43.9355202, -13.6309395, 26.3713303, -49.1033325, 57.4665718

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0963546, upper bound: 47.1028247
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0963546, upper bound: 47.1028247
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.6117496, 43.5632706, -7.3594851, 25.4020214, -38.0137711, 50.7615891
1: -14.9453526, 50.6414070, -8.4710608, 29.3300591, -44.2754059, 58.8794594
2: -15.5010452, 49.5077133, -9.0635443, 28.7711582, -44.2721977, 58.3804550
3: -22.8407478, 53.2817268, -12.9796400, 30.7849236, -53.6256714, 66.0219574
4: -24.8766441, 47.7960587, -14.4126072, 27.8498859, -52.7265320, 62.1362877

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1364445, upper bound: 47.1087050
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1364445, upper bound: 47.1087050
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.5850286, 40.2752228, -7.2959199, 25.2211227, -36.8061447, 47.3850479
1: -13.7641401, 46.7895126, -8.3935299, 29.1190872, -42.8832283, 54.9256401
2: -14.2358398, 45.7822647, -8.9715900, 28.5816212, -42.8174553, 54.5367851
3: -21.0448055, 49.2071724, -12.8681383, 30.5324993, -51.5773048, 61.7985878
4: -22.8478355, 44.1755142, -14.2740421, 27.6462288, -50.4940605, 58.3501549

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0963546, upper bound: 47.1552590
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591046, upper bound: 47.1552590
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.6719284, 43.7624931, -7.7365437, 26.7040882, -39.3760147, 51.3409729
1: -15.0196400, 50.8729935, -8.9201317, 30.8552704, -45.8749084, 59.5642395
2: -15.5719442, 49.7368774, -9.5132608, 30.2694721, -45.8414116, 59.0585060
3: -22.9502277, 53.5219193, -13.6697979, 32.3597908, -55.3100204, 66.9562378
4: -24.9835854, 48.0195274, -15.1328773, 29.2781792, -54.2617607, 63.0819855

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1550510, upper bound: 47.1550510
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1364445, upper bound: 47.1550510
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -10.7382069, 37.0713997, -7.3645077, 25.4187469, -36.1569481, 44.4359055
1: -12.6549892, 43.0521164, -8.4769840, 29.3495617, -42.0045433, 51.5290985
2: -13.2225256, 42.0720711, -9.0696068, 28.7901382, -42.0126648, 51.1416779
3: -19.3875694, 45.3159103, -12.9885550, 30.8053379, -50.1929092, 58.2877960
4: -21.2321262, 40.6393242, -14.4219141, 27.8682060, -49.1003342, 55.0612335

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0594387, upper bound: 47.0796095
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0449915, upper bound: 47.0481000
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -10.9528446, 37.6870155, -7.3645077, 25.4187469, -36.3715897, 45.0515213
1: -12.9132843, 43.7553520, -8.4769840, 29.3495617, -42.2628441, 52.2323380
2: -13.4681273, 42.7748718, -9.0696068, 28.7901382, -42.2582626, 51.8444786
3: -19.7512875, 46.0691261, -12.9885550, 30.8053379, -50.5566177, 59.0424500
4: -21.6068287, 41.3166008, -14.4219141, 27.8682060, -49.4750366, 55.7385101

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0594387, upper bound: 47.0806644
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0449915, upper bound: 47.0491549
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -10.7382069, 37.0713997, -7.7413502, 26.7201595, -37.4583664, 44.8127480
1: -12.6549892, 43.0521164, -8.9257545, 30.8739986, -43.5289879, 51.9778709
2: -13.2225256, 42.0720711, -9.5190830, 30.2876854, -43.5102081, 51.5911484
3: -19.3875694, 45.3159103, -13.6782970, 32.3794136, -51.7669830, 58.9791031
4: -21.2321262, 40.6393242, -15.1418896, 29.2957630, -50.5278893, 55.7812042

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0594387, upper bound: 47.0815132
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0449915, upper bound: 47.0478600
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.9528446, 37.6870155, -7.7413502, 26.7201595, -37.6730042, 45.4283600
1: -12.9132843, 43.7553520, -8.9257545, 30.8739986, -43.7872849, 52.6811066
2: -13.4681273, 42.7748718, -9.5190830, 30.2876854, -43.7558060, 52.2939491
3: -19.7512875, 46.0691261, -13.6782970, 32.3794136, -52.1306992, 59.7337570
4: -21.6068287, 41.3166008, -15.1418896, 29.2957630, -50.9025879, 56.4584846

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0594387, upper bound: 47.0825003
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0449915, upper bound: 47.0489149
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -12.6311121, 43.6649780, -5.8594179, 20.1327782, -32.7638893, 49.3424721
1: -14.9605846, 50.7695885, -6.6864014, 23.1041451, -38.0647278, 57.2094345
2: -15.5325613, 49.6204185, -7.2021346, 22.7496414, -38.2821999, 56.6250114
3: -22.8752594, 53.4159927, -10.1838274, 24.2136650, -47.0889244, 63.3710289
4: -24.9373646, 47.8990097, -11.3504562, 22.0592155, -46.9965820, 59.2347946

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0642804, upper bound: 47.0515950
time: 0.56 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0766683, upper bound: 47.0580147
time: 0.73 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0804335, upper bound: 47.0639848
time: 0.88 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0804335, upper bound: 47.0639848
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -10.9528446, 37.6870155, -5.8594179, 20.1327782, -31.0856171, 43.5464325
1: -12.9132843, 43.7553520, -6.6864014, 23.1041451, -36.0174294, 50.4417534
2: -13.4681273, 42.7748718, -7.2021346, 22.7496414, -36.2177544, 49.9770012
3: -19.7512875, 46.0691261, -10.1838274, 24.2136650, -43.9649467, 56.2529526
4: -21.6068287, 41.3166008, -11.3504562, 22.0592155, -43.6660461, 52.6670570

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0401747
time: 0.53 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0401747
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -12.6311121, 43.6649780, -6.2778120, 21.6416531, -34.2727661, 49.7596931
1: -14.9605846, 50.7695885, -7.1870689, 24.8781242, -39.8387070, 57.7101936
2: -15.5325613, 49.6204185, -7.7067957, 24.4813576, -40.0139198, 57.1267357
3: -22.8752594, 53.4159927, -10.9660168, 26.0270386, -48.9022980, 64.1546173
4: -24.9373646, 47.8990097, -12.1560593, 23.7112980, -48.6486626, 60.0421638

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.56 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0404536
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -10.9528446, 37.6870155, -6.2778120, 21.6416531, -32.5944977, 43.9648247
1: -12.9132843, 43.7553520, -7.1870689, 24.8781242, -37.7914085, 50.9424133
2: -13.4681273, 42.7748718, -7.7067957, 24.4813576, -37.9494781, 50.4816666
3: -19.7512875, 46.0691261, -10.9660168, 26.0270386, -45.7783279, 57.0351410
4: -21.6068287, 41.3166008, -12.1560593, 23.7112980, -45.3181267, 53.4726601

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.59 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.57 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 8.68 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1660596, upper bound: 47.1660596
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1660596, upper bound: 47.1749842
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1749842, upper bound: 47.1670978
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1749842, upper bound: 47.1797914
IS_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1701485, upper bound: 47.1652180
IS_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1722538, upper bound: 47.1652034
IS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1701479, upper bound: 47.1656178
IS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1720863, upper bound: 47.1655717
IS_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1673845, upper bound: 47.1660427
IS_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1673845, upper bound: 47.1679130
IS_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1673845, upper bound: 47.1673682
IS_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1673845, upper bound: 47.1679130
IS_A1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1678906, upper bound: 47.1660203
IS_A1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1678906, upper bound: 47.1678539
IS_A1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1678906, upper bound: 47.1672916
IS_A1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1678906, upper bound: 47.1678539
IS_A1_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1529227, upper bound: 47.1569773
IS_A1_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1658529, upper bound: 47.1600415
IS_A1_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1529227, upper bound: 47.1579637
IS_A1_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1664410, upper bound: 47.1609763
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1001542, upper bound: 47.0656287
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1001542, upper bound: 47.0658690
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1002428, upper bound: 47.0662547
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1002428, upper bound: 47.0665081
IS_A1_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0871541, upper bound: 47.0631217
IS_A1_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0871541, upper bound: 47.0631217
IS_A1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0871541, upper bound: 47.0632801
IS_A1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0871541, upper bound: 47.0632801
IS_A1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0873462, upper bound: 47.0636670
IS_A1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0873462, upper bound: 47.0636670
IS_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0873462, upper bound: 47.0637900
IS_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0873462, upper bound: 47.0637900
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0963546, upper bound: 47.1028247
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0963546, upper bound: 47.1028247
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1364445, upper bound: 47.1087050
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1364445, upper bound: 47.1087050
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0963546, upper bound: 47.1552590
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1591046, upper bound: 47.1552590
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1550510, upper bound: 47.1550510
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.1364445, upper bound: 47.1550510
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0594387, upper bound: 47.0796095
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0449915, upper bound: 47.0481000
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0594387, upper bound: 47.0806644
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0449915, upper bound: 47.0491549
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0594387, upper bound: 47.0815132
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0449915, upper bound: 47.0478600
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0594387, upper bound: 47.0825003
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0449915, upper bound: 47.0489149
IS_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0804335, upper bound: 47.0639848
IS_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0804335, upper bound: 47.0639848
IS_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0401747
IS_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0401747
IS_A2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
IS_A2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0404536
IS_A2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
IS_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.68
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.3339679, 12.4404860, -3.3324127, 12.4352713, -15.7692394, 15.7728968
1: -3.8323221, 14.3667116, -3.8295591, 14.3606071, -18.1929283, 18.1962662
2: -4.1936383, 13.9953842, -4.1917672, 13.9890413, -18.1826782, 18.1871510
3: -6.0318007, 15.0859957, -6.0276842, 15.0789547, -21.1107540, 21.1136799
4: -7.0685482, 13.2598248, -7.0652933, 13.2532454, -20.3217926, 20.3251190

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42

Time for candidate selection: 8.34 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 8
type: B, layer: 5, pos: 8
type: A, layer: 5, pos: 31
type: B, layer: 5, pos: 31
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 48
type: A, layer: 5, pos: 48
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 42
type: A, layer: 5, pos: 42
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 45
type: B, layer: 5, pos: 0
type: A, layer: 5, pos: 0
type: A, layer: 5, pos: 45
type: A, layer: 5, pos: 36
type: B, layer: 5, pos: 36
type: A, layer: 5, pos: 39
type: B, layer: 5, pos: 39
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 32
type: A, layer: 5, pos: 32
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 26
type: A, layer: 5, pos: 22
type: B, layer: 5, pos: 22
type: B, layer: 5, pos: 41
type: A, layer: 5, pos: 41
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 25

Time for candidate selection: 23.98 seconds

### Candidate
type: A, layer: 5, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1623598, upper bound: 47.1601367
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1623596, upper bound: 47.1623596
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.3339679, 12.4404860, -3.9541271, 14.3758478, -17.7098160, 16.3946114
1: -3.8323221, 14.3667116, -4.4822893, 16.6509914, -20.4833145, 18.8489933
2: -4.1936383, 13.9953842, -4.9704862, 16.1518536, -20.3454914, 18.9658699
3: -6.0318007, 15.0859957, -7.0345216, 17.5144024, -23.5461998, 22.1205177
4: -7.0685482, 13.2598248, -8.3142977, 15.3319921, -22.4005394, 21.5741234

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 18

Time for candidate selection: 8.72 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 31
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 31
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 48
type: A, layer: 5, pos: 48
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 42
type: A, layer: 5, pos: 8
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 42
type: B, layer: 5, pos: 0
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 45
type: B, layer: 5, pos: 36
type: A, layer: 5, pos: 0
type: A, layer: 5, pos: 45
type: A, layer: 5, pos: 36
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 39
type: A, layer: 5, pos: 39
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 26
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 32
type: B, layer: 5, pos: 32
type: B, layer: 5, pos: 22
type: A, layer: 5, pos: 41
type: B, layer: 5, pos: 41
type: A, layer: 5, pos: 22
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 25

Time for candidate selection: 24.84 seconds

### Candidate
type: B, layer: 5, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1623598, upper bound: 47.1734429
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1623596, upper bound: 47.1746913
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -3.9750445, 14.4460850, -3.3339679, 12.4404860, -16.4155293, 17.7800522
1: -4.5114536, 16.7320309, -3.8323221, 14.3667116, -18.8781605, 20.5643520
2: -4.9959445, 16.2338371, -4.1936383, 13.9953842, -18.9913292, 20.4274731
3: -7.0781264, 17.6019325, -6.0318007, 15.0859957, -22.1641216, 23.6337299
4: -8.3550758, 15.4140797, -7.0685482, 13.2598248, -21.6149006, 22.4826279

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1735273, upper bound: 47.1622251
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1749683, upper bound: 47.1665690
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1749683, upper bound: 47.1670978
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -3.9750445, 14.4460850, -3.9750445, 14.4460850, -18.4211292, 18.4211292
1: -4.5114536, 16.7320309, -4.5114536, 16.7320309, -21.2434826, 21.2434826
2: -4.9959445, 16.2338371, -4.9959445, 16.2338371, -21.2297802, 21.2297821
3: -7.0781264, 17.6019325, -7.0781264, 17.6019325, -24.6800594, 24.6800594
4: -8.3550758, 15.4140797, -8.3550758, 15.4140797, -23.7691555, 23.7691555

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1749683, upper bound: 47.1794904
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1749683, upper bound: 47.1795889
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -2.0597177, 8.2895155, -4.4739985, 15.8106270, -17.8703442, 12.7635136
1: -2.2967291, 9.6150331, -5.1033578, 18.2775612, -20.5742855, 14.7183895
2: -2.6521909, 9.2325039, -5.5320044, 17.7803707, -20.4325619, 14.7645082
3: -3.7075272, 10.1419230, -7.9086761, 19.1823235, -22.8898487, 18.0505962
4: -4.7798123, 8.5157490, -9.0676374, 16.9618435, -21.7416553, 17.5833855

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658175, upper bound: 47.1613822
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1652750, upper bound: 47.1613104
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -3.5526295, 13.1160755, -5.0363150, 17.6988392, -21.2514668, 18.1523876
1: -4.0184674, 15.2061739, -5.7646542, 20.4788914, -24.4973583, 20.9708271
2: -4.4789891, 14.7249899, -6.2293510, 19.9428024, -24.4217911, 20.9543324
3: -6.3371143, 15.9977407, -8.9515896, 21.4750729, -27.8121872, 24.9493294
4: -7.5789533, 13.9239416, -10.1371250, 19.0903397, -26.6692905, 24.0610657

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1722420, upper bound: 47.1646312
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1722420, upper bound: 47.1652034
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -2.3129969, 9.1597061, -5.9002466, 20.3143272, -22.6273232, 15.0599527
1: -2.5794778, 10.6262455, -6.7162523, 23.5529022, -26.1323795, 17.3424988
2: -2.9723115, 10.2174473, -7.2860713, 22.9001656, -25.8724766, 17.5035191
3: -4.1473298, 11.2064428, -10.3691654, 24.7531776, -28.9005070, 21.5756035
4: -5.2884049, 9.4641771, -11.8042498, 21.9775066, -27.2659092, 21.2684269

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1663015, upper bound: 47.1639381
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1680423, upper bound: 47.1633459
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -3.9372892, 14.3412886, -6.0500598, 20.8305531, -24.7678413, 20.3913383
1: -4.4667454, 16.6241646, -6.8981895, 24.1518002, -28.6185455, 23.5223522
2: -4.9550190, 16.1163063, -7.4732709, 23.4966431, -28.4516602, 23.5895748
3: -7.0154467, 17.5031776, -10.6497498, 25.3807373, -32.3961830, 28.1529274
4: -8.3169518, 15.2909412, -12.0981274, 22.5594177, -30.8763695, 27.3890686

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1720649, upper bound: 47.1650123
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1720649, upper bound: 47.1655717
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -5.1154051, 17.9620304, -3.3324127, 12.4352713, -17.5506744, 21.2944431
1: -5.8550353, 20.7860451, -3.8295591, 14.3606071, -20.2156391, 24.6156025
2: -6.3262277, 20.2429676, -4.1917672, 13.9890413, -20.3152676, 24.4347343
3: -9.0907803, 21.7931328, -6.0276842, 15.0789547, -24.1697350, 27.8208141
4: -10.2859144, 19.3818359, -7.0652933, 13.2532454, -23.5391560, 26.4471264

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 42

Time for candidate selection: 8.71 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1667487, upper bound: 47.1646066
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1667487, upper bound: 47.1653778
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -5.1154051, 17.9620304, -3.9541271, 14.3758478, -19.4912529, 21.9161568
1: -5.8550353, 20.7860451, -4.4822893, 16.6509914, -22.5060234, 25.2683315
2: -6.3262277, 20.2429676, -4.9704862, 16.1518536, -22.4780807, 25.2134533
3: -9.0907803, 21.7931328, -7.0345216, 17.5144024, -26.6051826, 28.8276539
4: -10.2859144, 19.3818359, -8.3142977, 15.3319921, -25.6179066, 27.6961327

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1656852, upper bound: 47.1694659
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0833333, mid=0.0833333, abs_max=50.8192024230957
rel_dist={4: [-47.18088696914194, 47.18088696914194]}

## Binary search (step 2) starts
Candidate diff: 0.0416667


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1725005, upper bound: 47.1680812
time: 0.52 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678244, upper bound: 47.1678244
time: 0.90 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.58 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 4, lower bound: -47.1725005, upper bound: 47.1680812
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 4, lower bound: -47.1678244, upper bound: 47.1678244

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.3289332, 15.6493454, -5.8106618, 20.5728836, -24.9018173, 21.4600010
1: -4.9227071, 18.1375618, -6.6639028, 23.8333817, -28.7560883, 24.8014641
2: -5.4355350, 17.6055393, -7.2380991, 23.2434902, -28.6790257, 24.8436375
3: -7.7107296, 19.0719624, -10.3406324, 25.0188999, -32.7296295, 29.4125938
4: -9.0420084, 16.7446327, -11.7947159, 22.2706470, -31.3126507, 28.5393486

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678244, upper bound: 47.1678244
time: 0.50 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678244, upper bound: 47.1678244
time: 0.86 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -13.8669882, 47.8264809, -8.3915739, 28.9772472, -42.8442345, 56.1235580
1: -16.4424095, 55.6123199, -9.7024679, 33.5246391, -49.9670448, 65.1655273
2: -17.0157623, 54.3669052, -10.3182182, 32.8587723, -49.8745346, 64.5716019
3: -25.0840302, 58.4735527, -14.8692255, 35.1634789, -60.2475090, 73.1976547
4: -27.2452755, 52.4822922, -16.4273300, 31.7668839, -59.0121613, 68.9096222

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1405356, upper bound: 47.1280381
time: 0.90 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678184, upper bound: 47.1678184
time: 0.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.83 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.83
Output dim: 4, lower bound: -47.1678244, upper bound: 47.1678244
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.83
Output dim: 4, lower bound: -47.1678244, upper bound: 47.1678244
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.83
Output dim: 4, lower bound: -47.1405356, upper bound: 47.1280381
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.83
Output dim: 4, lower bound: -47.1678184, upper bound: 47.1678184

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -4.3289332, 15.6493454, -4.3289332, 15.6493454, -19.9782753, 19.9782753
1: -4.9227071, 18.1375618, -4.9227071, 18.1375618, -23.0602684, 23.0602684
2: -5.4355350, 17.6055393, -5.4355350, 17.6055393, -23.0410748, 23.0410748
3: -7.7107296, 19.0719624, -7.7107296, 19.0719624, -26.7826920, 26.7826920
4: -9.0420084, 16.7446327, -9.0420084, 16.7446327, -25.7866364, 25.7866364

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1711260, upper bound: 47.1650464
time: 0.49 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712532, upper bound: 47.1678131
time: 0.49 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -4.3289332, 15.6493454, -12.4547701, 42.7791710, -46.9673615, 28.1041126
1: -4.9227071, 18.1375618, -14.6426277, 49.7638359, -54.5080185, 32.7801895
2: -5.4355350, 17.6055393, -15.3634157, 48.6813507, -53.9675903, 32.9689560
3: -7.7107296, 19.0719624, -22.4968510, 52.4056053, -59.9649010, 41.5688057
4: -9.0420084, 16.7446327, -24.6993065, 47.0198441, -56.0618477, 41.4439354

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1719655, upper bound: 47.1680424
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712532, upper bound: 47.1678131
time: 0.47 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -13.7583323, 47.4687347, -7.9303670, 27.4180527, -41.1763840, 55.3001823
1: -16.3092003, 55.1972885, -9.1398411, 31.7070980, -48.0162964, 64.1820831
2: -16.8879375, 53.9563103, -9.7707472, 31.0626278, -47.9505653, 63.6116104
3: -24.8880939, 58.0422478, -14.0170469, 33.2817345, -58.1698303, 71.9057617
4: -27.0526638, 52.0813980, -15.5677376, 30.0443630, -57.0970230, 67.6491318

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0528613, upper bound: 47.0622476
time: 0.86 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0392994, upper bound: 47.0401747
time: 0.50 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -13.8669882, 47.8264809, -8.2426720, 28.4884987, -42.3554878, 55.9748383
1: -16.4424095, 55.6123199, -9.5107641, 32.9592133, -49.4016151, 64.9745102
2: -17.0157623, 54.3669052, -10.1405535, 32.2973747, -49.3131294, 64.3928986
3: -25.0840302, 58.4735527, -14.5850163, 34.5719376, -59.6559677, 72.9110107
4: -27.2452755, 52.4822922, -16.1585922, 31.2181377, -58.4634132, 68.6408844

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1280381, upper bound: 47.1405356
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1280381, upper bound: 47.1678184
time: 0.88 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.55 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 4, lower bound: -47.1711260, upper bound: 47.1650464
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 4, lower bound: -47.1712532, upper bound: 47.1678131
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 4, lower bound: -47.1719655, upper bound: 47.1680424
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 4, lower bound: -47.1712532, upper bound: 47.1678131
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 4, lower bound: -47.0528613, upper bound: 47.0622476
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 4, lower bound: -47.0392994, upper bound: 47.0401747
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 4, lower bound: -47.1280381, upper bound: 47.1405356
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 4, lower bound: -47.1280381, upper bound: 47.1678184

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -4.2146969, 15.2717171, -4.0885248, 14.8552752, -19.0699692, 19.3602409
1: -4.7881989, 17.7000198, -4.6401472, 17.2174129, -22.0056095, 22.3401680
2: -5.2951260, 17.1733284, -5.1397719, 16.6970844, -21.9922085, 22.3131008
3: -7.5067816, 18.6168194, -7.2818685, 18.1143341, -25.6211147, 25.8986874
4: -8.8308811, 16.3196030, -8.5971518, 15.8510361, -24.6819172, 24.9167557

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679817, upper bound: 47.1718355
time: 0.57 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1685196, upper bound: 47.1717081
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -4.1830254, 15.1674328, -6.2036738, 21.3862705, -25.5692959, 21.3711071
1: -4.7386937, 17.5865173, -7.0725803, 24.8031158, -29.5418091, 24.6590919
2: -5.2617745, 17.0533199, -7.6664505, 24.1312027, -29.3929768, 24.7197704
3: -7.4378581, 18.5015926, -10.9312239, 26.0610123, -33.4988708, 29.4328156
4: -8.7825384, 16.1980629, -12.4161844, 23.1557178, -31.9382553, 28.6142464

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679025, upper bound: 47.1673740
time: 0.50 seconds

## Relational analysis of IS_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678801, upper bound: 47.1678801
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.0885248, 14.8552752, -12.3251219, 42.3561783, -46.3020973, 27.1803951
1: -4.6401472, 17.2174129, -14.4855070, 49.2731934, -53.7342911, 31.7029190
2: -5.1397719, 16.6970844, -15.2041855, 48.1930389, -53.1839561, 31.9012642
3: -7.2818685, 18.1143341, -22.2595272, 51.8911285, -59.0219765, 40.3738556
4: -8.5971518, 15.8510361, -24.4598408, 46.5368500, -55.1340027, 40.3108749

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1688618, upper bound: 47.1649255
time: 0.49 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1690761, upper bound: 47.1626326
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.2036738, 21.3862705, -12.2810822, 42.2073250, -48.2654991, 33.6673508
1: -7.0725803, 24.8031158, -14.4278364, 49.1068459, -55.9918480, 39.2309532
2: -7.6664505, 24.1312027, -15.1595078, 48.0220413, -55.5422668, 39.2906990
3: -10.9312239, 26.0610123, -22.1767178, 51.7235794, -62.4985199, 48.2377205
4: -12.4161844, 23.1557178, -24.3907242, 46.3758621, -58.7897415, 47.5464401

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1676607, upper bound: 47.1645941
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1677092, upper bound: 47.1621612
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -13.4104872, 46.2880859, -7.2901583, 25.1731739, -38.5836601, 53.4398880
1: -15.8958492, 53.8199883, -8.3876095, 29.0662975, -44.9621468, 62.0063858
2: -16.4679928, 52.6118355, -8.9787626, 28.5090256, -44.9770164, 61.4271889
3: -24.2693310, 56.6036873, -12.8539200, 30.5082779, -54.7776031, 69.2490005
4: -26.3953934, 50.7858238, -14.2844210, 27.5918007, -53.9871864, 65.0172119

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0392994, upper bound: 47.0401747
time: 0.55 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0392994, upper bound: 47.0401747
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -11.9435806, 41.1477356, -5.8067050, 19.9594975, -31.9030781, 46.9158249
1: -14.1057940, 47.8238564, -6.6242385, 22.9049606, -37.0107536, 54.3804741
2: -14.6797533, 46.7404366, -7.1367569, 22.5504322, -37.2301865, 53.8340530
3: -21.5681705, 50.3344994, -10.0902910, 24.0051079, -45.5732803, 60.3627968
4: -23.5693779, 45.1126366, -11.2514009, 21.8641338, -45.4335060, 56.3640366

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0390189, upper bound: 47.0390189
time: 0.54 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0390189, upper bound: 47.0401747
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.4713326, 46.5864258, -8.2426720, 28.4884987, -41.9598312, 54.7258110
1: -15.9442806, 54.1922150, -9.5107641, 32.9592133, -48.9034958, 63.5469475
2: -16.5558205, 52.9386673, -10.1405535, 32.2973747, -48.8531914, 62.9571495
3: -24.3609676, 56.9773254, -14.5850163, 34.5719376, -58.9329071, 71.4084320
4: -26.5542812, 51.0858421, -16.1585922, 31.2181377, -57.7724190, 67.2444305

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0526159, upper bound: 47.0616738
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0390189, upper bound: 47.0392994
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.5907269, 46.9298134, -8.2426720, 28.4884987, -42.0792236, 55.0718307
1: -16.1015148, 54.5770988, -9.5107641, 32.9592133, -49.0607300, 63.9351044
2: -16.6906662, 53.3377571, -10.1405535, 32.2973747, -48.9880371, 63.3580971
3: -24.5839233, 57.3939667, -14.5850163, 34.5719376, -59.1558533, 71.8251801
4: -26.7548733, 51.4779167, -16.1585922, 31.2181377, -57.9730110, 67.6365051

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0526159, upper bound: 47.0627326
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0390189, upper bound: 47.0404536
time: 0.57 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.32 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -47.1679817, upper bound: 47.1718355
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -47.1685196, upper bound: 47.1717081
IS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -47.1679025, upper bound: 47.1673740
IS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -47.1678801, upper bound: 47.1678801
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -47.1688618, upper bound: 47.1649255
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -47.1690761, upper bound: 47.1626326
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -47.1676607, upper bound: 47.1645941
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -47.1677092, upper bound: 47.1621612
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -47.0392994, upper bound: 47.0401747
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -47.0392994, upper bound: 47.0401747
IS_A2_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 4, lower bound: -47.0390189, upper bound: 47.0390189
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -47.0390189, upper bound: 47.0401747
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -47.0526159, upper bound: 47.0616738
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 4, lower bound: -47.0390189, upper bound: 47.0392994
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -47.0526159, upper bound: 47.0627326
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -47.0390189, upper bound: 47.0404536

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -3.4220200, 12.7278166, -3.4281158, 12.7430267, -16.1650467, 16.1559334
1: -3.9328542, 14.7019968, -3.8733091, 14.7726078, -18.7054577, 18.5753059
2: -4.2998037, 14.3202457, -4.3229051, 14.3020554, -18.6018562, 18.6431503
3: -6.1852064, 15.4322748, -6.1196384, 15.5253620, -21.7105675, 21.5519142
4: -7.2305393, 13.5755987, -7.3346567, 13.5008507, -20.7313900, 20.9102554

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1666277, upper bound: 47.1718355
time: 0.53 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1666277, upper bound: 47.1718355
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -4.0888171, 14.8227129, -4.0374079, 14.6729851, -18.7618027, 18.8601208
1: -4.6447906, 17.1685905, -4.5820498, 17.0015030, -21.6462898, 21.7506390
2: -5.1376886, 16.6643734, -5.0756617, 16.4906082, -21.6282959, 21.7400322
3: -7.2826281, 18.0583935, -7.1908154, 17.8869991, -25.1696281, 25.2492085
4: -8.5683126, 15.8390875, -8.4899530, 15.6560240, -24.2243366, 24.3290348

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1676777, upper bound: 47.1716246
time: 0.72 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1676777, upper bound: 47.1716895
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -3.5239389, 13.0709705, -5.1154051, 17.9620304, -21.4859695, 18.1863747
1: -3.9745758, 15.1590223, -5.8550353, 20.7860451, -24.7606201, 21.0140572
2: -4.4460807, 14.6776476, -6.3262277, 20.2429676, -24.6890488, 21.0038757
3: -6.2808967, 15.9306126, -9.0907803, 21.7931328, -28.0740261, 25.0213871
4: -7.5253258, 13.8661270, -10.2859144, 19.3818359, -26.9071579, 24.1520405

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679025, upper bound: 47.1673740
time: 0.70 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679025, upper bound: 47.1672811
time: 0.49 seconds

## BFS IS instance: IS_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -4.1237731, 14.9602098, -6.1029205, 21.0067959, -25.1305676, 21.0631294
1: -4.6710749, 17.3412628, -6.9584203, 24.3565216, -29.0275936, 24.2996788
2: -5.1880627, 16.8186550, -7.5369096, 23.6974564, -28.8855190, 24.3555641
3: -7.3325124, 18.2438469, -10.7420139, 25.5904827, -32.9229965, 28.9858608
4: -8.6613445, 15.9750090, -12.1946831, 22.7528343, -31.4141788, 28.1696892

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678801, upper bound: 47.1678801
time: 0.56 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678801, upper bound: 47.1678801
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.4281158, 12.7430267, -11.3390789, 39.2123833, -42.4687996, 24.0821056
1: -3.8733091, 14.7726078, -13.3673964, 45.5792427, -49.2507210, 28.1399975
2: -4.3229051, 14.3020554, -13.9812632, 44.6204453, -48.7749901, 28.2833176
3: -6.1196384, 15.5253620, -20.5429268, 47.9838142, -53.9200211, 36.0682907
4: -7.3346567, 13.5008507, -22.5018806, 43.0724602, -50.3796844, 36.0027313

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1641602, upper bound: 47.1618498
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1641602, upper bound: 47.1649095
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.0374079, 14.6729851, -12.1550159, 41.7759438, -45.6835022, 26.8280010
1: -4.5820498, 17.0015030, -14.2844391, 48.5951767, -53.0122261, 31.2859421
2: -5.0756617, 16.4906082, -14.9928541, 47.5295715, -52.4688530, 31.4834614
3: -7.1908154, 17.8869991, -21.9504490, 51.1776886, -58.2321777, 39.8374405
4: -8.4899530, 15.6560240, -24.1252365, 45.8941765, -54.3841286, 39.7812614

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1653400, upper bound: 47.1613774
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1690636, upper bound: 47.1626166
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.3982167, 18.8519669, -11.2689018, 38.9660873, -44.1976967, 30.1208687
1: -6.1418462, 21.8543777, -13.2733088, 45.3050079, -51.2447510, 35.1276855
2: -6.6933846, 21.2404251, -13.8988552, 44.3341751, -50.8602219, 35.1392784
3: -9.5385046, 22.9489231, -20.4056816, 47.6985970, -57.0466156, 43.3546028
4: -10.9023895, 20.3329468, -22.3787975, 42.7908516, -53.6530571, 42.7117462

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660015, upper bound: 47.1583368
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1652441, upper bound: 47.1622654
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.1613002, 21.2259884, -12.1077480, 41.6145439, -47.6417084, 33.3337250
1: -7.0246091, 24.6136894, -14.2230263, 48.4144096, -55.2629662, 38.8367157
2: -7.6115317, 23.9487057, -14.9443674, 47.3437805, -54.8213196, 38.8930702
3: -10.8516417, 25.8620415, -21.8621311, 50.9957657, -61.7059860, 47.7241516
4: -12.3226185, 22.9857864, -24.0507812, 45.7188568, -58.0414734, 47.0365562

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1613796, upper bound: 47.1459859
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1672030, upper bound: 47.1621612
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1672030, upper bound: 47.1621612
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -12.8014841, 44.2135429, -7.2901583, 25.1731739, -37.9746475, 51.3308182
1: -15.1710854, 51.4003944, -8.3876095, 29.0662975, -44.2373810, 59.5418434
2: -15.7324066, 50.2490005, -8.9787626, 28.5090256, -44.2414246, 59.0248871
3: -23.1827412, 54.0752983, -12.8539200, 30.5082779, -53.6910095, 66.6774597
4: -25.2400856, 48.5107803, -14.2844210, 27.5918007, -52.8318863, 62.7148209

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0526159, upper bound: 47.0611886
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0526159, upper bound: 47.0622476
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -11.1373444, 38.2923012, -7.2901583, 25.1731739, -36.3105164, 45.5724678
1: -13.1465988, 44.4529190, -8.3876095, 29.0662975, -42.2128944, 52.7966652
2: -13.6858282, 43.4720726, -8.9787626, 28.5090256, -42.1948471, 52.4225388
3: -20.0959721, 46.7951050, -12.8539200, 30.5082779, -50.6042404, 59.5805244
4: -21.9348602, 42.0024948, -14.2844210, 27.5918007, -49.5266571, 56.2869148

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0526159, upper bound: 47.0611886
time: 0.55 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0526159, upper bound: 47.0622476
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -11.7598743, 40.5483437, -5.8067050, 19.9594975, -31.7193661, 46.3107071
1: -13.8741169, 47.1327438, -6.6242385, 22.9049606, -36.7790756, 53.6860123
2: -14.4617310, 46.0488815, -7.1367569, 22.5504322, -37.0121613, 53.1373711
3: -21.2244987, 49.6113434, -10.0902910, 24.0051079, -45.2296066, 59.6339340
4: -23.2413349, 44.4314270, -11.2514009, 21.8641338, -45.1054649, 55.6828270

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0390189, upper bound: 47.0401747
time: 0.50 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0390189, upper bound: 47.0401747
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.1404905, 45.4638252, -7.5286980, 26.0219975, -39.1624832, 52.8502312
1: -15.5517979, 52.8822136, -8.6753902, 30.0637894, -45.6155853, 61.3565712
2: -16.1584015, 51.6599693, -9.2641144, 29.4918823, -45.6502762, 60.7552910
3: -23.7729378, 55.6084023, -13.3020353, 31.5350533, -55.3079872, 68.7021637
4: -25.9288158, 49.8555794, -14.7533865, 28.5245056, -54.4533157, 64.5524673

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0401747, upper bound: 47.0392994
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0401747, upper bound: 47.0392994
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.2433701, 45.7510605, -7.5286980, 26.0219975, -39.2653656, 53.1406174
1: -15.6887703, 53.2022285, -8.6753902, 30.0637894, -45.7525520, 61.6795006
2: -16.2713604, 51.9957275, -9.2641144, 29.4918823, -45.7632446, 61.0923843
3: -23.9663200, 55.9574738, -13.3020353, 31.5350533, -55.5013695, 69.0506363
4: -26.0981941, 50.1850204, -14.7533865, 28.5245056, -54.6226997, 64.8839188

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -11.7598743, 40.5483437, -6.1101933, 21.0805111, -32.8403854, 46.6141930
1: -13.8741169, 47.1327438, -6.9892278, 24.2255592, -38.0996704, 54.0517235
2: -14.4617310, 46.0488815, -7.5007567, 23.8385811, -38.3003120, 53.4993286
3: -21.2244987, 49.6113434, -10.6646528, 25.3490257, -46.5735245, 60.2099075
4: -23.2413349, 44.4314270, -11.8421850, 23.0870533, -46.3283806, 56.2736092

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.96 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.89 seconds
IS_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.1666277, upper bound: 47.1718355
IS_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.1666277, upper bound: 47.1718355
IS_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.1676777, upper bound: 47.1716246
IS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.1676777, upper bound: 47.1716895
IS_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.1679025, upper bound: 47.1673740
IS_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.1679025, upper bound: 47.1672811
IS_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.1678801, upper bound: 47.1678801
IS_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.1678801, upper bound: 47.1678801
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.1641602, upper bound: 47.1618498
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.1641602, upper bound: 47.1649095
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.1653400, upper bound: 47.1613774
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.1690636, upper bound: 47.1626166
IS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.1660015, upper bound: 47.1583368
IS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.1652441, upper bound: 47.1622654
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.1672030, upper bound: 47.1621612
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.1672030, upper bound: 47.1621612
IS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.0526159, upper bound: 47.0611886
IS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.0526159, upper bound: 47.0622476
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.0526159, upper bound: 47.0611886
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.0526159, upper bound: 47.0622476
IS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.0390189, upper bound: 47.0401747
IS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.0390189, upper bound: 47.0401747
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.0401747, upper bound: 47.0392994
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.0401747, upper bound: 47.0392994
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536

## BFS IS instance: IS_A1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -3.3339679, 12.4404860, -3.4281158, 12.7430267, -16.0769939, 15.8686018
1: -3.8323221, 14.3667116, -3.8733091, 14.7726078, -18.6049290, 18.2400150
2: -4.1936383, 13.9953842, -4.3229051, 14.3020554, -18.4956894, 18.3182850
3: -6.0318007, 15.0859957, -6.1196384, 15.5253620, -21.5571613, 21.2056351
4: -7.0685482, 13.2598248, -7.3346567, 13.5008507, -20.5693989, 20.5944824

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1618268, upper bound: 47.1701926
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660596, upper bound: 47.1660427
time: 0.84 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660596, upper bound: 47.1718355
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -5.0065460, 17.5910816, -3.4281158, 12.7430267, -17.7495708, 21.0191975
1: -5.7073598, 20.3670120, -3.8733091, 14.7726078, -20.4799633, 24.2403221
2: -6.1937943, 19.7996254, -4.3229051, 14.3020554, -20.4958477, 24.1225300
3: -8.8686228, 21.3635139, -6.1196384, 15.5253620, -24.3939857, 27.4831524
4: -10.0971718, 18.9318256, -7.3346567, 13.5008507, -23.5980225, 26.2664814

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1657505, upper bound: 47.1717536
time: 0.56 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1666277, upper bound: 47.1718280
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.0096622, 14.5677414, -2.9563463, 11.2644262, -15.2740879, 17.5240860
1: -4.5505328, 16.8711758, -3.3138306, 13.0760107, -17.6265430, 20.1850052
2: -5.0420771, 16.3712273, -3.7716575, 12.6109381, -17.6530113, 20.1428852
3: -7.1410270, 17.7479038, -5.2604923, 13.7730932, -20.9141197, 23.0083961
4: -8.4243069, 15.5501623, -6.5178967, 11.8244390, -20.2487450, 22.0680542

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1676777, upper bound: 47.1716246
time: 0.77 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1676777, upper bound: 47.1716246
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.0888171, 14.8227129, -3.9151268, 14.2869701, -18.3757877, 18.7378387
1: -4.6447906, 17.1685905, -4.4264760, 16.5557575, -21.2005463, 21.5950661
2: -5.1376886, 16.6643734, -4.9285703, 16.0426903, -21.1803780, 21.5929432
3: -7.2826281, 18.0583935, -6.9592967, 17.4172993, -24.6999264, 25.0176888
4: -8.5683126, 15.8390875, -8.2715330, 15.2093029, -23.7776146, 24.1106186

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1685196, upper bound: 47.1716895
time: 0.99 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1685196, upper bound: 47.1716895
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -3.4281158, 12.7430267, -5.1154051, 17.9620304, -21.3901463, 17.8584309
1: -3.8733091, 14.7726078, -5.8550353, 20.7860451, -24.6593533, 20.6276379
2: -4.3229051, 14.3020554, -6.3262277, 20.2429676, -24.5658703, 20.6282787
3: -6.1196384, 15.5253620, -9.0907803, 21.7931328, -27.9127712, 24.6161423
4: -7.3346567, 13.5008507, -10.2859144, 19.3818359, -26.7164917, 23.7867641

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660427, upper bound: 47.1673740
time: 0.51 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660427, upper bound: 47.1673740
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -5.3727970, 18.7546120, -5.1154051, 17.9620304, -23.3348274, 23.8700161
1: -6.1069884, 21.7433834, -5.8550353, 20.7860451, -26.8930302, 27.5984154
2: -6.6701646, 21.1229858, -6.3262277, 20.2429676, -26.9131317, 27.4492130
3: -9.4929228, 22.8417301, -9.0907803, 21.7931328, -31.2860527, 31.9325027
4: -10.8596001, 20.2271366, -10.2859144, 19.3818359, -30.2414360, 30.5130501

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660427, upper bound: 47.1672811
time: 0.52 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660427, upper bound: 47.1672811
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -4.0237517, 14.6279736, -6.1029205, 21.0067959, -25.0305481, 20.7308941
1: -4.5655317, 16.9498672, -6.9584203, 24.3565216, -28.9220543, 23.9082870
2: -5.0590835, 16.4392452, -7.5369096, 23.6974564, -28.7565403, 23.9761543
3: -7.1657147, 17.8330097, -10.7420139, 25.5904827, -32.7561989, 28.5750237
4: -8.4649611, 15.6054611, -12.1946831, 22.7528343, -31.2177963, 27.8001423

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1614269, upper bound: 47.1661886
time: 0.91 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1656327, upper bound: 47.1656327
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -6.1382365, 21.1266117, -6.1029205, 21.0067959, -27.1450310, 27.2295322
1: -6.9997768, 24.5037270, -6.9584203, 24.3565216, -31.3562984, 31.4621449
2: -7.5966086, 23.8292351, -7.5369096, 23.6974564, -31.2940655, 31.3661442
3: -10.8245420, 25.7712440, -10.7420139, 25.5904827, -36.4150200, 36.5132599
4: -12.2938814, 22.8886871, -12.1946831, 22.7528343, -35.0467148, 35.0833626

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660203, upper bound: 47.1678801
time: 0.83 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660203, upper bound: 47.1678434
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -2.4417143, 9.6244287, -11.2644472, 38.9717712, -41.2355042, 20.8888741
1: -2.7288489, 11.1765013, -13.2768497, 45.2996483, -47.8161240, 24.4533501
2: -3.1381807, 10.7503157, -13.8920174, 44.3429832, -47.3121567, 24.6423283
3: -4.3771992, 11.7672005, -20.4076080, 47.6906586, -51.8825874, 32.1748085
4: -5.5399389, 10.0040054, -22.3665333, 42.8002853, -48.3069229, 32.3705368

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B1_A1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1641308, upper bound: 47.1612117
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619034, upper bound: 47.1565296
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1612370, upper bound: 47.1599586
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -3.3240724, 12.4184580, -11.3390789, 39.2123833, -42.3650131, 23.7575378
1: -3.7446253, 14.4000082, -13.3673964, 45.5792427, -49.1210365, 27.7673969
2: -4.2003059, 13.9290867, -13.9812632, 44.6204453, -48.6522217, 27.9103508
3: -5.9305983, 15.1347513, -20.5429268, 47.9838142, -53.7271652, 35.6776772
4: -7.1521735, 13.1321392, -22.5018806, 43.0724602, -50.1895714, 35.6340179

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1657799, upper bound: 47.1646157
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1657799, upper bound: 47.1649095
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -2.9563463, 11.2644262, -12.0820847, 41.5413666, -44.3559952, 23.3465099
1: -3.3138306, 13.0760107, -14.1954575, 48.3234940, -51.4642143, 27.2714634
2: -3.7716575, 12.6109381, -14.9051723, 47.2590904, -50.8895798, 27.5161076
3: -5.2604923, 13.7730932, -21.8175621, 50.8920517, -56.0139961, 35.5906563
4: -6.5178967, 11.8244390, -23.9927139, 45.6274414, -52.1453323, 35.8171539

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1601575, upper bound: 47.1560236
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1653400, upper bound: 47.1602638
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1653400, upper bound: 47.1613774
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.9151268, 14.2869701, -12.1550159, 41.7759438, -45.5617714, 26.4419861
1: -4.4264760, 16.5557575, -14.2844391, 48.5951767, -52.8556175, 30.8401966
2: -4.9285703, 16.0426903, -14.9928541, 47.5295715, -52.3212357, 31.0355453
3: -6.9592967, 17.4172993, -21.9504490, 51.1776886, -57.9967537, 39.3677483
4: -8.2715330, 15.2093029, -24.1252365, 45.8941765, -54.1657104, 39.3345413

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1632996, upper bound: 47.1464104
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1586049, upper bound: 47.1464104
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -5.3982167, 18.8519669, -11.0046148, 38.1502113, -43.3852501, 29.8565826
1: -6.1418462, 21.8543777, -12.9681625, 44.3551941, -50.2992096, 34.8225365
2: -6.6933846, 21.2404251, -13.5840826, 43.4019852, -49.9317780, 34.8245010
3: -9.5385046, 22.9489231, -19.9488029, 46.7068100, -56.0572968, 42.8977280
4: -10.9023895, 20.3329468, -21.9015255, 41.8782425, -52.7421722, 42.2344742

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1654590, upper bound: 47.1583238
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1654590, upper bound: 47.1583368
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -5.3548989, 18.7212582, -11.0896463, 38.4692917, -43.6799393, 29.8108997
1: -6.0919976, 21.7032413, -13.0808887, 44.7344627, -50.6521988, 34.7841301
2: -6.6427188, 21.0915051, -13.6857357, 43.7794037, -50.2780609, 34.7772408
3: -9.4656258, 22.7910290, -20.1225662, 47.1047707, -56.4023743, 42.9135857
4: -10.8268251, 20.1860104, -22.0672684, 42.2448196, -53.0475388, 42.2532806

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1647019, upper bound: 47.1622523
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1647019, upper bound: 47.1622654
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.1154051, 17.9620304, -12.1077480, 41.6145439, -46.5153503, 30.0697784
1: -5.8550353, 20.7860451, -14.2230263, 48.4144096, -54.0009308, 35.0090637
2: -6.3262277, 20.2429676, -14.9443674, 47.3437805, -53.4412231, 35.1873322
3: -9.0907803, 21.7931328, -21.8621311, 50.9957657, -59.8362885, 43.6552544
4: -10.2859144, 19.3818359, -24.0507812, 45.7188568, -55.9228592, 43.4326172

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1608271, upper bound: 47.1600229
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1608271, upper bound: 47.1594554
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.1029205, 21.0067959, -12.1077480, 41.6145439, -47.5830193, 33.1145401
1: -6.9584203, 24.3565216, -14.2230263, 48.4144096, -55.1959419, 38.5795479
2: -7.5369096, 23.6974564, -14.9443674, 47.3437805, -54.7493706, 38.6418228
3: -10.7420139, 25.5904827, -21.8621311, 50.9957657, -61.6006012, 47.4526062
4: -12.1946831, 22.7528343, -24.0507812, 45.7188568, -57.9135399, 46.8036079

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1654590, upper bound: 47.1562138
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1645872, upper bound: 47.1594684
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -12.5650520, 43.4993477, -7.2901583, 25.1731739, -37.7382278, 50.6129112
1: -14.8664980, 50.5900497, -8.3876095, 29.0662975, -43.9327965, 58.7329597
2: -15.4648190, 49.4217529, -8.9787626, 28.5090256, -43.9738388, 58.1967888
3: -22.7446213, 53.2128296, -12.8539200, 30.5082779, -53.2528992, 65.8155060
4: -24.8334503, 47.7021942, -14.2844210, 27.5918007, -52.4252472, 61.9038467

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0826313, upper bound: 47.0734362
time: 0.52 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0903079, upper bound: 47.0903079
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -12.6311121, 43.6649780, -7.2901583, 25.1731739, -37.8042870, 50.7796974
1: -14.9605846, 50.7695885, -8.3876095, 29.0662975, -44.0268784, 58.9122314
2: -15.5325613, 49.6204185, -8.9787626, 28.5090256, -44.0415840, 58.3953819
3: -22.8752594, 53.4159927, -12.8539200, 30.5082779, -53.3835335, 66.0163803
4: -24.9373646, 47.8990097, -14.2844210, 27.5918007, -52.5291672, 62.1011658

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0826313, upper bound: 47.0734362
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0903079, upper bound: 47.0986127
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -10.7382069, 37.0713997, -7.2901583, 25.1731739, -35.9113808, 44.3484344
1: -12.6549892, 43.0521164, -8.3876095, 29.0662975, -41.7212868, 51.3891029
2: -13.2225256, 42.0720711, -8.9787626, 28.5090256, -41.7315445, 51.0186310
3: -19.3875694, 45.3159103, -12.8539200, 30.5082779, -49.8958473, 58.0943222
4: -21.2321262, 40.6393242, -14.2844210, 27.5918007, -48.8239288, 54.9237442

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0516842, upper bound: 47.0602696
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0449308, upper bound: 47.0480896
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -10.9528446, 37.6870155, -7.2901583, 25.1731739, -36.1260185, 44.9615746
1: -12.9132843, 43.7553520, -8.3876095, 29.0662975, -41.9795837, 52.0965309
2: -13.4681273, 42.7748718, -8.9787626, 28.5090256, -41.9771385, 51.7208900
3: -19.7512875, 46.0691261, -12.8539200, 30.5082779, -50.2595634, 58.8493614
4: -21.6068287, 41.3166008, -14.2844210, 27.5918007, -49.1986313, 55.6010208

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0516842, upper bound: 47.0602696
time: 1.77 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0449308, upper bound: 47.0491549
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -12.4401388, 43.0676994, -5.8067050, 19.9594975, -32.3996353, 48.6895943
1: -14.7418556, 50.0859337, -6.6242385, 22.9049606, -37.6468163, 56.4608955
2: -15.3110914, 48.9577904, -7.1367569, 22.5504322, -37.8615227, 55.8892021
3: -22.5544796, 52.7031403, -10.0902910, 24.0051079, -46.5595856, 62.5545616
4: -24.6029949, 47.2582817, -11.2514009, 21.8641338, -46.4671288, 58.4761963

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0392994, upper bound: 47.0401747
time: 0.59 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0392994, upper bound: 47.0401747
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -10.9528446, 37.6870155, -5.8067050, 19.9594975, -30.9123402, 43.4937134
1: -12.9132843, 43.7553520, -6.6242385, 22.9049606, -35.8182449, 50.3795891
2: -13.4681273, 42.7748718, -7.1367569, 22.5504322, -36.0185585, 49.9116287
3: -19.7512875, 46.0691261, -10.0902910, 24.0051079, -43.7563934, 56.1594124
4: -21.6068287, 41.3166008, -11.2514009, 21.8641338, -43.4709625, 52.5680008

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0392994, upper bound: 47.0401747
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0392994, upper bound: 47.0401747
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.5650520, 43.4993477, -7.5286980, 26.0219975, -38.5870514, 50.8523216
1: -14.8664980, 50.5900497, -8.6753902, 30.0637894, -44.9302864, 59.0216866
2: -15.4648190, 49.4217529, -9.2641144, 29.4918823, -44.9566994, 58.4793816
3: -22.7446213, 53.2128296, -13.3020353, 31.5350533, -54.2796745, 66.2651596
4: -24.8334503, 47.7021942, -14.7533865, 28.5245056, -53.3579559, 62.3733025

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0516842, upper bound: 47.0607712
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0450639, upper bound: 47.0478490
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -10.7382069, 37.0713997, -7.5286980, 26.0219975, -36.7602043, 44.5878487
1: -12.6549892, 43.0521164, -8.6753902, 30.0637894, -42.7187805, 51.6778259
2: -13.2225256, 42.0720711, -9.2641144, 29.4918823, -42.7144051, 51.3012276
3: -19.3875694, 45.3159103, -13.3020353, 31.5350533, -50.9226227, 58.5439835
4: -21.2321262, 40.6393242, -14.7533865, 28.5245056, -49.7566299, 55.3927116

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0516842, upper bound: 47.0607712
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0450639, upper bound: 47.0478490
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.6311121, 43.6649780, -7.5286980, 26.0219975, -38.6531105, 51.0205879
1: -14.9605846, 50.7695885, -8.6753902, 30.0637894, -45.0243683, 59.2022514
2: -15.5325613, 49.6204185, -9.2641144, 29.4918823, -45.0244446, 58.6779747
3: -22.8752594, 53.4159927, -13.3020353, 31.5350533, -54.4103127, 66.4660339
4: -24.9373646, 47.8990097, -14.7533865, 28.5245056, -53.4618683, 62.5706177

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0542003, upper bound: 47.0617503
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0452159, upper bound: 47.0489149
time: 0.93 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 7.35 seconds
IS_A1_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1660596, upper bound: 47.1660427
IS_A1_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1660596, upper bound: 47.1718355
IS_A1_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1657505, upper bound: 47.1717536
IS_A1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1666277, upper bound: 47.1718280
IS_A1_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1676777, upper bound: 47.1716246
IS_A1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1676777, upper bound: 47.1716246
IS_A1_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1685196, upper bound: 47.1716895
IS_A1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1685196, upper bound: 47.1716895
IS_A1_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1660427, upper bound: 47.1673740
IS_A1_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1660427, upper bound: 47.1673740
IS_A1_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1660427, upper bound: 47.1672811
IS_A1_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1660427, upper bound: 47.1672811
IS_A1_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1614269, upper bound: 47.1661886
IS_A1_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1656327, upper bound: 47.1656327
IS_A1_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1660203, upper bound: 47.1678801
IS_A1_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1660203, upper bound: 47.1678434
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1619034, upper bound: 47.1565296
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1612370, upper bound: 47.1599586
IS_A1_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1657799, upper bound: 47.1646157
IS_A1_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1657799, upper bound: 47.1649095
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1653400, upper bound: 47.1602638
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1653400, upper bound: 47.1613774
IS_A1_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1632996, upper bound: 47.1464104
IS_A1_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1586049, upper bound: 47.1464104
IS_A1_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1654590, upper bound: 47.1583238
IS_A1_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1654590, upper bound: 47.1583368
IS_A1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1647019, upper bound: 47.1622523
IS_A1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1647019, upper bound: 47.1622654
IS_A1_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1608271, upper bound: 47.1600229
IS_A1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1608271, upper bound: 47.1594554
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1654590, upper bound: 47.1562138
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.1645872, upper bound: 47.1594684
IS_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.0826313, upper bound: 47.0734362
IS_A2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.0903079, upper bound: 47.0903079
IS_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.0826313, upper bound: 47.0734362
IS_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.0903079, upper bound: 47.0986127
IS_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.0516842, upper bound: 47.0602696
IS_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.0449308, upper bound: 47.0480896
IS_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.0516842, upper bound: 47.0602696
IS_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.0449308, upper bound: 47.0491549
IS_A2_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.0392994, upper bound: 47.0401747
IS_A2_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.0392994, upper bound: 47.0401747
IS_A2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.0392994, upper bound: 47.0401747
IS_A2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.0392994, upper bound: 47.0401747
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.0516842, upper bound: 47.0607712
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.0450639, upper bound: 47.0478490
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.0516842, upper bound: 47.0607712
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.0450639, upper bound: 47.0478490
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.0542003, upper bound: 47.0617503
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.35
Output dim: 4, lower bound: -47.0452159, upper bound: 47.0489149
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.35
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.35
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.35
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0416667, mid=0.0416667, abs_max=50.8192024230957
rel_dist={4: [-47.180735877080295, 47.18073587708028]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1134.32 seconds
