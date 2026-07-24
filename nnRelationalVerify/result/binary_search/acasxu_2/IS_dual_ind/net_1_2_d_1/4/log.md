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
execution time: IAR + LP analysis = 1.71 + 1.93 = 3.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -47.1809221, upper bound: 47.1809221


# Binary Search by BASE starts (time budget: 1196.37 seconds, max iter: 100)

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
Binary search time: 64.97 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1131.39 seconds

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1787782, upper bound: 47.1690313
time: 0.84 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.84 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.83 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.83
Output dim: 4, lower bound: -47.1787782, upper bound: 47.1690313
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.83
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.3289332, 15.6493454, -8.8739548, 30.5588493, -34.8877831, 24.5232983
1: -4.9227071, 18.1375618, -10.2695408, 35.3635216, -40.2862282, 28.4071026
2: -5.4355350, 17.6055393, -10.8957434, 34.6628456, -40.0983772, 28.5012817
3: -7.7107296, 19.0719624, -15.7225132, 37.0795212, -44.7902489, 34.7944641
4: -9.0420084, 16.7446327, -17.3079681, 33.5112419, -42.5532455, 34.0525932

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.46 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.83 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -13.8669882, 47.8264809, -8.8639011, 30.5256615, -44.3926430, 56.5944176
1: -16.4424095, 55.6123199, -10.2576370, 35.3249588, -51.7673607, 65.7206268
2: -17.0157623, 54.3669052, -10.8837032, 34.6249809, -51.6407394, 65.1386719
3: -25.0840302, 58.4735527, -15.7045870, 37.0393791, -62.1234093, 74.0339813
4: -27.2452755, 52.4822922, -17.2896118, 33.4746056, -60.7198792, 69.7719040

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.55 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.82 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.82
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.82
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.82
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.82
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -4.3289332, 15.6493454, -4.3289332, 15.6493454, -19.9782753, 19.9782753
1: -4.9227071, 18.1375618, -4.9227071, 18.1375618, -23.0602684, 23.0602684
2: -5.4355350, 17.6055393, -5.4355350, 17.6055393, -23.0410748, 23.0410748
3: -7.7107296, 19.0719624, -7.7107296, 19.0719624, -26.7826920, 26.7826920
4: -9.0420084, 16.7446327, -9.0420084, 16.7446327, -25.7866364, 25.7866364

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1785033, upper bound: 47.1687905
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1787754, upper bound: 47.1690043
time: 0.55 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -4.3289332, 15.6493454, -13.8669882, 47.8264809, -52.0300293, 29.5163288
1: -4.9227071, 18.1375618, -16.4424095, 55.6123199, -60.3761864, 34.5799713
2: -5.4355350, 17.6055393, -17.0157623, 54.3669052, -59.6714478, 34.6212921
3: -7.7107296, 19.0719624, -25.0840302, 58.4735527, -66.0495911, 44.1559906
4: -9.0420084, 16.7446327, -27.2452755, 52.4822922, -61.5242958, 43.9899063

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1785033, upper bound: 47.1687905
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1787754, upper bound: 47.1690043
time: 0.96 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -13.8669882, 47.8264809, -4.3289332, 15.6493454, -29.5163288, 52.0300331
1: -16.4424095, 55.6123199, -4.9227071, 18.1375618, -34.5799713, 60.3761864
2: -17.0157623, 54.3669052, -5.4355350, 17.6055393, -34.6212921, 59.6714478
3: -25.0840302, 58.4735527, -7.7107296, 19.0719624, -44.1559906, 66.0495911
4: -27.2452755, 52.4822922, -9.0420084, 16.7446327, -43.9899063, 61.5242958

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1646659, upper bound: 47.1619752
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1622348
time: 0.50 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -13.8669882, 47.8264809, -13.8669882, 47.8264809, -61.5810852, 61.5810928
1: -16.4424095, 55.6123199, -16.4424095, 55.6123199, -71.8540726, 71.8540726
2: -17.0157623, 54.3669052, -17.0157623, 54.3669052, -71.2083588, 71.2083588
3: -25.0840302, 58.4735527, -25.0840302, 58.4735527, -83.3268814, 83.3268738
4: -27.2452755, 52.4822922, -27.2452755, 52.4822922, -79.6028137, 79.6028061

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1646659, upper bound: 47.1619752
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1622348
time: 0.49 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.78 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 4, lower bound: -47.1785033, upper bound: 47.1687905
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 4, lower bound: -47.1787754, upper bound: 47.1690043
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 4, lower bound: -47.1785033, upper bound: 47.1687905
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 4, lower bound: -47.1787754, upper bound: 47.1690043
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 4, lower bound: -47.1646659, upper bound: 47.1619752
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1622348
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 4, lower bound: -47.1646659, upper bound: 47.1619752
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1622348

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.1957455, 15.1547375, -4.3289332, 15.6493454, -19.8450909, 19.4836674
1: -4.7651372, 17.5528717, -4.9227071, 18.1375618, -22.9026985, 22.4755783
2: -5.2724891, 17.0362396, -5.4355350, 17.6055393, -22.8780251, 22.4717751
3: -7.4683981, 18.4647770, -7.7107296, 19.0719624, -26.5403595, 26.1755066
4: -8.7734575, 16.2064075, -9.0420084, 16.7446327, -25.5180874, 25.2484131

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1801766, upper bound: 47.1801766
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1801766, upper bound: 47.1803884
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.2849174, 15.4173374, -4.3289332, 15.6493454, -19.9342556, 19.7462673
1: -4.8534427, 17.8700066, -4.9227071, 18.1375618, -22.9910049, 22.7927132
2: -5.3807850, 17.3343678, -5.4355350, 17.6055393, -22.9863243, 22.7699032
3: -7.5899143, 18.7914963, -7.7107296, 19.0719624, -26.6618767, 26.5022259
4: -8.9349260, 16.4891949, -9.0420084, 16.7446327, -25.6795559, 25.5312004

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1803884, upper bound: 47.1804113
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1803884, upper bound: 47.1806231
time: 0.47 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.1957455, 15.1547375, -13.8669882, 47.8264809, -51.8964386, 29.0217209
1: -4.7651372, 17.5528717, -16.4424095, 55.6123199, -60.2181740, 33.9952774
2: -5.2724891, 17.0362396, -17.0157623, 54.3669052, -59.5081711, 34.0519943
3: -7.4683981, 18.4647770, -25.0840302, 58.4735527, -65.8062592, 43.5488052
4: -8.7734575, 16.2064075, -27.2452755, 52.4822922, -61.2557487, 43.4516830

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1774904, upper bound: 47.1654456
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1780975, upper bound: 47.1628426
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.2849174, 15.4173374, -13.8669882, 47.8264809, -51.9881248, 29.2843208
1: -4.8534427, 17.8700066, -16.4424095, 55.6123199, -60.3079033, 34.3124084
2: -5.3807850, 17.3343678, -17.0157623, 54.3669052, -59.6177826, 34.3501282
3: -7.5899143, 18.7914963, -25.0840302, 58.4735527, -65.9300919, 43.8755264
4: -8.9349260, 16.4891949, -27.2452755, 52.4822922, -61.4172096, 43.7344704

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1776654, upper bound: 47.1656594
time: 0.49 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1782519, upper bound: 47.1630621
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.7235136, 44.1252480, -4.2184191, 15.2935438, -28.0170517, 48.2138176
1: -15.1153173, 51.2670212, -4.7932100, 17.7262554, -32.8415718, 55.9023972
2: -15.6082811, 50.1621857, -5.2993908, 17.2012253, -32.8095055, 55.3275528
3: -23.0708389, 53.8880844, -7.5149736, 18.6378288, -41.7086678, 61.2588501
4: -24.9927807, 48.4072723, -8.8317080, 16.3487091, -41.3414917, 57.2389793

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1654456, upper bound: 47.1774904
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1656594, upper bound: 47.1776654
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.6422262, 47.0618324, -4.3289332, 15.6493454, -29.2915688, 51.2752151
1: -16.1728191, 54.7214432, -4.9227071, 18.1375618, -34.3103790, 59.4951324
2: -16.7433243, 53.4933968, -5.4355350, 17.6055393, -34.3488579, 58.8068466
3: -24.6783352, 57.5419312, -7.7107296, 19.0719624, -43.7502899, 65.1278000
4: -26.8195095, 51.6375656, -9.0420084, 16.7446327, -43.5641327, 60.6795731

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1628426, upper bound: 47.1780975
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1630621, upper bound: 47.1782519
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.7235136, 44.1252480, -13.7310543, 47.3840904, -59.9850845, 57.7445107
1: -15.1153173, 51.2670212, -16.2827091, 55.0976639, -69.9942017, 67.3527603
2: -15.6082811, 50.1621857, -16.8521748, 53.8637619, -69.2893524, 66.8405914
3: -23.0708389, 53.8880844, -24.8444958, 57.9328957, -80.7515869, 78.4950409
4: -24.9927807, 48.4072723, -26.9897575, 51.9940720, -76.8434601, 75.2671432

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1646462, upper bound: 47.1607082
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1644444, upper bound: 47.1619553
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.6422262, 47.0618324, -13.8669882, 47.8264809, -61.3553696, 60.8262749
1: -16.1728191, 54.7214432, -16.4424095, 55.6123199, -71.5860977, 70.9730225
2: -16.7433243, 53.4933968, -17.0157623, 54.3669052, -70.9353943, 70.3437500
3: -24.6783352, 57.5419312, -25.0840302, 58.4735527, -82.9250488, 82.4050903
4: -26.8195095, 51.6375656, -27.2452755, 52.4822922, -79.1803589, 78.7645264

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.58 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.91 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -47.1801766, upper bound: 47.1801766
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -47.1801766, upper bound: 47.1803884
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -47.1803884, upper bound: 47.1804113
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -47.1803884, upper bound: 47.1806231
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -47.1774904, upper bound: 47.1654456
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -47.1780975, upper bound: 47.1628426
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -47.1776654, upper bound: 47.1656594
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -47.1782519, upper bound: 47.1630621
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -47.1654456, upper bound: 47.1774904
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -47.1656594, upper bound: 47.1776654
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -47.1628426, upper bound: 47.1780975
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -47.1630621, upper bound: 47.1782519
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -47.1646462, upper bound: 47.1607082
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -47.1644444, upper bound: 47.1619553
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.1957455, 15.1547375, -4.1957455, 15.1547375, -19.3504810, 19.3504829
1: -4.7651372, 17.5528717, -4.7651372, 17.5528717, -22.3180084, 22.3180084
2: -5.2724891, 17.0362396, -5.2724891, 17.0362396, -22.3087254, 22.3087254
3: -7.4683981, 18.4647770, -7.4683981, 18.4647770, -25.9331741, 25.9331741
4: -8.7734575, 16.2064075, -8.7734575, 16.2064075, -24.9798641, 24.9798641

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0938498, upper bound: 47.1574752
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1799102, upper bound: 47.1799103
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.1957455, 15.1547375, -4.2849174, 15.4173374, -19.6130829, 19.4396477
1: -4.7651372, 17.5528717, -4.8534427, 17.8700066, -22.6351433, 22.4063129
2: -5.2724891, 17.0362396, -5.3807850, 17.3343678, -22.6068497, 22.4170246
3: -7.4683981, 18.4647770, -7.5899143, 18.7914963, -26.2598953, 26.0546913
4: -8.7734575, 16.2064075, -8.9349260, 16.4891949, -25.2626514, 25.1413326

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0938498, upper bound: 47.1574926
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1799102, upper bound: 47.1801665
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.2849174, 15.4173374, -4.1957455, 15.1547375, -19.4396477, 19.6130829
1: -4.8534427, 17.8700066, -4.7651372, 17.5528717, -22.4063148, 22.6351433
2: -5.3807850, 17.3343678, -5.2724891, 17.0362396, -22.4170246, 22.6068516
3: -7.5899143, 18.7914963, -7.4683981, 18.4647770, -26.0546913, 26.2598953
4: -8.9349260, 16.4891949, -8.7734575, 16.2064075, -25.1413326, 25.2626514

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1770036, upper bound: 47.1680614
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.2849174, 15.4173374, -4.2849174, 15.4173374, -19.7022495, 19.7022495
1: -4.8534427, 17.8700066, -4.8534427, 17.8700066, -22.7234478, 22.7234478
2: -5.3807850, 17.3343678, -5.3807850, 17.3343678, -22.7151527, 22.7151527
3: -7.5899143, 18.7914963, -7.5899143, 18.7914963, -26.3814106, 26.3814106
4: -8.9349260, 16.4891949, -8.9349260, 16.4891949, -25.4241199, 25.4241199

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1773324, upper bound: 47.1723043
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1712896
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.0841393, 14.7974739, -12.7235136, 44.1252480, -48.0790558, 27.5209885
1: -4.6350403, 17.1381226, -15.1153173, 51.2670212, -55.7437782, 32.2534409
2: -5.1350250, 16.6290226, -15.6082811, 50.1621857, -55.1628952, 32.2373009
3: -7.2715898, 18.0266895, -23.0708389, 53.8880844, -61.0144234, 41.0975266
4: -8.5603867, 15.8077469, -24.9927807, 48.4072723, -56.9675751, 40.8005295

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0886168, upper bound: 47.1041045
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1767857, upper bound: 47.1627924
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.1957455, 15.1547375, -13.6422262, 47.0618324, -51.1416283, 28.7969608
1: -4.7651372, 17.5528717, -16.1728191, 54.7214432, -59.3371201, 33.7256927
2: -5.2724891, 17.0362396, -16.7433243, 53.4933968, -58.6435699, 33.7795601
3: -7.4683981, 18.4647770, -24.6783352, 57.5419312, -64.8844604, 43.1431122
4: -8.7734575, 16.2064075, -26.8195095, 51.6375656, -60.4110222, 43.0259171

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0916935, upper bound: 47.1376392
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1774295, upper bound: 47.1598867
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.1661615, 15.0392208, -12.7235136, 44.1252480, -48.1639748, 27.7627316
1: -4.7162590, 17.4329357, -15.1153173, 51.2670212, -55.8262062, 32.5482521
2: -5.2362127, 16.9053440, -15.6082811, 50.1621857, -55.2650375, 32.5136223
3: -7.3827748, 18.3315296, -23.0708389, 53.8880844, -61.1278458, 41.4023590
4: -8.7125006, 16.0695858, -24.9927807, 48.4072723, -57.1197739, 41.0623665

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1771196, upper bound: 47.1655771
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1676764, upper bound: 47.1641860
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.2849174, 15.4173374, -13.6422262, 47.0618324, -51.2333107, 29.0595627
1: -4.8534427, 17.8700066, -16.1728191, 54.7214432, -59.4268494, 34.0428238
2: -5.3807850, 17.3343678, -16.7433243, 53.4933968, -58.7531815, 34.0776901
3: -7.5899143, 18.7914963, -24.6783352, 57.5419312, -65.0083084, 43.4698334
4: -8.9349260, 16.4891949, -26.8195095, 51.6375656, -60.5724869, 43.3087044

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1772581, upper bound: 47.1629814
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1677345, upper bound: 47.1615719
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -12.7235136, 44.1252480, -4.0841393, 14.7974739, -27.5209866, 48.0790558
1: -15.1153173, 51.2670212, -4.6350403, 17.1381226, -32.2534409, 55.7437782
2: -15.6082811, 50.1621857, -5.1350250, 16.6290226, -32.2373009, 55.1628952
3: -23.0708389, 53.8880844, -7.2715898, 18.0266895, -41.0975266, 61.0144234
4: -24.9927807, 48.4072723, -8.5603867, 15.8077469, -40.8005295, 56.9675751

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1563547, upper bound: 47.1441724
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0639228, upper bound: 47.1478454
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1627923, upper bound: 47.1767859
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.7235136, 44.1252480, -4.1661615, 15.0392208, -27.7627316, 48.1639748
1: -15.1153173, 51.2670212, -4.7162590, 17.4329357, -32.5482521, 55.8262062
2: -15.6082811, 50.1621857, -5.2362127, 16.9053440, -32.5136261, 55.2650375
3: -23.0708389, 53.8880844, -7.3827748, 18.3315296, -41.4023590, 61.1278458
4: -24.9927807, 48.4072723, -8.7125006, 16.0695858, -41.0623665, 57.1197739

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0642882, upper bound: 47.1489449
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1630635, upper bound: 47.1770027
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.6422262, 47.0618324, -4.1957455, 15.1547375, -28.7969608, 51.1416283
1: -16.1728191, 54.7214432, -4.7651372, 17.5528717, -33.7256927, 59.3371201
2: -16.7433243, 53.4933968, -5.2724891, 17.0362396, -33.7795563, 58.6435699
3: -24.6783352, 57.5419312, -7.4683981, 18.4647770, -43.1431122, 64.8844681
4: -26.8195095, 51.6375656, -8.7734575, 16.2064075, -43.0259171, 60.4110222

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1632567
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630569
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.6422262, 47.0618324, -4.2849174, 15.4173374, -29.0595627, 51.2333107
1: -16.1728191, 54.7214432, -4.8534427, 17.8700066, -34.0428238, 59.4268494
2: -16.7433243, 53.4933968, -5.3807850, 17.3343678, -34.0776901, 58.7531815
3: -24.6783352, 57.5419312, -7.5899143, 18.7914963, -43.4698334, 65.0083008
4: -26.8195095, 51.6375656, -8.9349260, 16.4891949, -43.3087006, 60.5724869

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603529, upper bound: 47.1679343
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1615719, upper bound: 47.1677345
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -12.7235136, 44.1252480, -13.4117832, 46.3328934, -58.9287491, 57.4214172
1: -15.1153173, 51.2670212, -15.8963299, 53.8752975, -68.7635498, 66.9644928
2: -15.6082811, 50.1621857, -16.4676399, 52.6564484, -68.0749512, 66.4530411
3: -23.0708389, 53.8880844, -24.2661896, 56.6583176, -79.4693069, 77.9157486
4: -24.9927807, 48.4072723, -26.4058723, 50.8118477, -75.6560745, 74.6810989

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0785863, upper bound: 47.0629028
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0636258, upper bound: 47.1369244
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619395, upper bound: 47.1575481
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.6965485, 44.0343628, -15.5656719, 53.0211678, -65.5507812, 59.4728165
1: -15.0817461, 51.1625671, -18.3832092, 61.6654167, -76.4654388, 69.3424911
2: -15.5761366, 50.0577354, -19.0133343, 60.2645607, -75.6009903, 68.8879700
3: -23.0210171, 53.7791901, -27.9824181, 64.8170547, -87.5238190, 81.5109711
4: -24.9443798, 48.3052177, -30.2938747, 58.2179794, -82.9802933, 78.4549332

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0634854, upper bound: 47.1383174
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1617991, upper bound: 47.1589411
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.6422262, 47.0618324, -12.7235136, 44.1252480, -57.6537743, 59.6676521
1: -16.1728191, 54.7214432, -15.1153173, 51.2670212, -67.2438583, 69.6207123
2: -16.7433243, 53.4933968, -15.6082811, 50.1621857, -66.7305374, 68.9216690
3: -24.6783352, 57.5419312, -23.0708389, 53.8880844, -78.3313370, 80.3638000
4: -26.8195095, 51.6375656, -24.9927807, 48.4072723, -75.0983505, 76.4888458

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1607074, upper bound: 47.1622151
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619545, upper bound: 47.1620134
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.6422262, 47.0618324, -13.6422262, 47.0618324, -60.6005516, 60.6005516
1: -16.1728191, 54.7214432, -16.1728191, 54.7214432, -70.7050476, 70.7050476
2: -16.7433243, 53.4933968, -16.7433243, 53.4933968, -70.0707855, 70.0707855
3: -24.6783352, 57.5419312, -24.6783352, 57.5419312, -82.0032578, 82.0032578
4: -26.8195095, 51.6375656, -26.8195095, 51.6375656, -78.3420792, 78.3420792

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1607074, upper bound: 47.1622151
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1607074, upper bound: 47.1620134
time: 0.89 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.31 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.0938498, upper bound: 47.1574752
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1799102, upper bound: 47.1799103
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.0938498, upper bound: 47.1574926
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1799102, upper bound: 47.1801665
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1770036, upper bound: 47.1680614
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1773324, upper bound: 47.1723043
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1712896
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.0886168, upper bound: 47.1041045
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1767857, upper bound: 47.1627924
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.0916935, upper bound: 47.1376392
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1774295, upper bound: 47.1598867
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1771196, upper bound: 47.1655771
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1676764, upper bound: 47.1641860
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1772581, upper bound: 47.1629814
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1677345, upper bound: 47.1615719
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.0639228, upper bound: 47.1478454
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1627923, upper bound: 47.1767859
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.0642882, upper bound: 47.1489449
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1630635, upper bound: 47.1770027
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1632567
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630569
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1603529, upper bound: 47.1679343
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1615719, upper bound: 47.1677345
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.0636258, upper bound: 47.1369244
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1619395, upper bound: 47.1575481
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.0634854, upper bound: 47.1383174
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1617991, upper bound: 47.1589411
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1607074, upper bound: 47.1622151
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1619545, upper bound: 47.1620134
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1607074, upper bound: 47.1622151
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 4, lower bound: -47.1607074, upper bound: 47.1620134

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.9499731, 14.3560972, -4.1532617, 15.0145960, -18.9645672, 18.5093555
1: -4.4719973, 16.6005650, -4.7136397, 17.3895206, -21.8615170, 21.3141994
2: -4.9373617, 16.1121292, -5.2183580, 16.8748741, -21.8122368, 21.3304844
3: -7.0062442, 17.3973083, -7.3882208, 18.2894421, -25.2956867, 24.7855301
4: -8.1801634, 15.2708197, -8.6852016, 16.0456009, -24.2257652, 23.9560204

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0767665, upper bound: 47.0767665
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0767665, upper bound: 47.1574752
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.0273848, 14.6178732, -4.1957455, 15.1547375, -19.1821175, 18.8136177
1: -4.5628810, 16.9356003, -4.7651372, 17.5528717, -22.1157532, 21.7007370
2: -5.0649624, 16.4214878, -5.2724891, 17.0362396, -22.1012020, 21.6939754
3: -7.1684437, 17.8108006, -7.4683981, 18.4647770, -25.6332188, 25.2791977
4: -8.4600706, 15.5963697, -8.7734575, 16.2064075, -24.6664772, 24.3698273

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1574752, upper bound: 47.0938498
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1574752, upper bound: 47.1799104
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.9499731, 14.3560972, -4.2424736, 15.2772923, -19.2272644, 18.5985661
1: -4.4719973, 16.6005650, -4.8010621, 17.7070503, -22.1790466, 21.4016266
2: -4.9373617, 16.1121292, -5.3265758, 17.1728706, -22.1102333, 21.4387035
3: -7.0062442, 17.3973083, -7.5088124, 18.6158581, -25.6221008, 24.9061203
4: -8.1801634, 15.2708197, -8.8464460, 16.3278599, -24.5080223, 24.1172657

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0684806, upper bound: 47.1359632
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0939348, upper bound: 47.1574926
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0932470, upper bound: 47.1473520
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.0273848, 14.6178732, -4.2849174, 15.4173374, -19.4447193, 18.9027901
1: -4.5628810, 16.9356003, -4.8534427, 17.8700066, -22.4328880, 21.7890415
2: -5.0649624, 16.4214878, -5.3807850, 17.3343678, -22.3993301, 21.8022728
3: -7.1684437, 17.8108006, -7.5899143, 18.7914963, -25.9599380, 25.4007149
4: -8.4600706, 15.5963697, -8.9349260, 16.4891949, -24.9492645, 24.5312958

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1652310, upper bound: 47.1768055
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1640355, upper bound: 47.1673695
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.0656214, 14.6907520, -4.1957455, 15.1547375, -19.2203579, 18.8864975
1: -4.5916448, 17.0283775, -4.7651372, 17.5528717, -22.1445160, 21.7935143
2: -5.1096640, 16.4983921, -5.2724891, 17.0362396, -22.1459045, 21.7708778
3: -7.1945591, 17.9123573, -7.4683981, 18.4647770, -25.6593342, 25.3807564
4: -8.5250015, 15.6656189, -8.7734575, 16.2064075, -24.7314091, 24.4390717

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.1992731, 21.2122402, -4.1746674, 15.0849533, -21.2842216, 25.3869076
1: -7.0349321, 24.6053638, -4.7387280, 17.4728680, -24.5077991, 29.3440914
2: -7.6394072, 23.9174843, -5.2474179, 16.9559536, -24.5953598, 29.1649017
3: -10.8425484, 25.8447590, -7.4292226, 18.3820171, -29.2245636, 33.2739792
4: -12.3402233, 22.9567204, -8.7357931, 16.1271706, -28.4673939, 31.6925125

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.0656214, 14.6907520, -4.2849174, 15.4173374, -19.4829559, 18.9756699
1: -4.5916448, 17.0283775, -4.8534427, 17.8700066, -22.4616489, 21.8818188
2: -5.1096640, 16.4983921, -5.3807850, 17.3343678, -22.4440308, 21.8791771
3: -7.1945591, 17.9123573, -7.5899143, 18.7914963, -25.9860535, 25.5022717
4: -8.5250015, 15.6656189, -8.9349260, 16.4891949, -25.0141964, 24.6005402

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1712896
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1712896
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.1992731, 21.2122402, -4.2647104, 15.3515139, -21.5507851, 25.4769516
1: -7.0349321, 24.6053638, -4.8285580, 17.7947578, -24.8296871, 29.4339218
2: -7.6394072, 23.9174843, -5.3569169, 17.2589989, -24.8984070, 29.2744007
3: -10.8425484, 25.8447590, -7.5529947, 18.7136173, -29.5561638, 33.3977470
4: -12.3402233, 22.9567204, -8.8992605, 16.4146023, -28.7548256, 31.8559780

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1712896
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1712896
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.8461647, 14.0256577, -12.6813402, 43.9865456, -47.6999359, 26.7069969
1: -4.3524222, 16.2163048, -15.0630436, 51.1049118, -55.2946396, 31.2793465
2: -4.8096867, 15.7374773, -15.5547285, 50.0014229, -54.6751328, 31.2922058
3: -6.8268933, 16.9921341, -22.9906216, 53.7125206, -60.3911552, 39.9827461
4: -7.9824967, 14.9052515, -24.9042377, 48.2471924, -56.2213898, 39.8094902

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0608190, upper bound: 47.0765287
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0704967, upper bound: 47.0475230
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0704967, upper bound: 47.1041045
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -3.9194047, 14.2739916, -12.7235136, 44.1252480, -47.9127922, 26.9975052
1: -4.4381156, 16.5363541, -15.1153173, 51.2670212, -55.5457268, 31.6516705
2: -4.9312482, 16.0302601, -15.6082811, 50.1621857, -54.9595718, 31.6385422
3: -6.9790220, 17.3885593, -23.0708389, 53.8880844, -60.7225647, 40.4593964
4: -8.2539015, 15.2128563, -24.9927807, 48.4072723, -56.6611214, 40.2056351

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1415090, upper bound: 47.1532238
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1478454, upper bound: 47.0639228
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1478454, upper bound: 47.1627924
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.9499731, 14.3560972, -13.5999355, 46.9237175, -50.7551193, 27.9560318
1: -4.4719973, 16.6005650, -16.1204357, 54.5599098, -58.8767891, 32.7210007
2: -4.9373617, 16.1121292, -16.6896610, 53.3332977, -58.1467896, 32.8017883
3: -7.0062442, 17.3973083, -24.5980644, 57.3666306, -64.2437363, 41.9953728
4: -8.1801634, 15.2708197, -26.7311268, 51.4776268, -59.6577911, 42.0019455

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0666638, upper bound: 47.1119218
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0743056, upper bound: 47.0595056
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0704967, upper bound: 47.1376392
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.0273848, 14.6178732, -13.6422262, 47.0618324, -50.9717560, 28.2600994
1: -4.5628810, 16.9356003, -16.1728191, 54.7214432, -59.1337662, 33.1084175
2: -5.0649624, 16.4214878, -16.7433243, 53.4933968, -58.4366455, 33.1648064
3: -7.1684437, 17.8108006, -24.6783352, 57.5419312, -64.5852966, 42.4891357
4: -8.4600706, 15.5963697, -26.8195095, 51.6375656, -60.0976334, 42.4158669

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603719, upper bound: 47.1558452
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1602370, upper bound: 47.1573137
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -3.9491742, 14.3233261, -12.7235136, 44.1252480, -47.9466629, 27.0468330
1: -4.4580026, 16.6038208, -15.1153173, 51.2670212, -55.5690880, 31.7191372
2: -4.9676933, 16.0823917, -15.6082811, 50.1621857, -54.9978790, 31.6906738
3: -6.9926357, 17.4643440, -23.0708389, 53.8880844, -60.7393951, 40.5351830
4: -8.3071480, 15.2576637, -24.9927807, 48.4072723, -56.7144203, 40.2504425

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1489449, upper bound: 47.0642090
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1764554, upper bound: 47.1629686
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.0817451, 20.8432217, -12.6965485, 44.0343628, -49.9758377, 33.5397720
1: -6.8995233, 24.1751728, -15.0817461, 51.1625671, -57.8885384, 39.2569199
2: -7.4971895, 23.4976387, -15.5761366, 50.0577354, -57.4231339, 39.0737762
3: -10.6402817, 25.3906555, -23.0210171, 53.7791901, -64.2694778, 48.4116745
4: -12.1190033, 22.5459404, -24.9443798, 48.3052177, -60.4145889, 47.4903183

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1446770, upper bound: 47.0635409
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1656945, upper bound: 47.1615503
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.0656214, 14.6907520, -13.6422262, 47.0618324, -51.0134773, 28.3329773
1: -4.5916448, 17.0283775, -16.1728191, 54.7214432, -59.1664124, 33.2011948
2: -5.1096640, 16.4983921, -16.7433243, 53.4933968, -58.4837379, 33.2417145
3: -7.1945591, 17.9123573, -24.6783352, 57.5419312, -64.6148529, 42.5906906
4: -8.5250015, 15.6656189, -26.8195095, 51.6375656, -60.1625595, 42.4851265

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1677345, upper bound: 47.1603529
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1677345, upper bound: 47.1615719
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.1992731, 21.2122402, -13.6181641, 46.9813080, -53.0538254, 34.8304062
1: -7.0349321, 24.6053638, -16.1430454, 54.6287346, -61.4982758, 40.7484093
2: -7.6394072, 23.9174843, -16.7145805, 53.4008980, -60.9215927, 40.6320648
3: -10.8425484, 25.8447590, -24.6341267, 57.4452667, -68.1580429, 50.4788857
4: -12.3402233, 22.9567204, -26.7762661, 51.5471916, -63.8874130, 49.7329865

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1677345, upper bound: 47.1603529
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1677345, upper bound: 47.1615719
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.6461382, 43.8475800, -4.0424170, 14.6602983, -27.3064365, 47.7475433
1: -15.0353851, 50.8989334, -4.5843019, 16.9781914, -32.0135765, 55.3073082
2: -15.4672871, 49.8316040, -5.0815153, 16.4710999, -31.9383869, 54.7631874
3: -22.8897457, 53.4193306, -7.1924739, 17.8543015, -40.7440414, 60.4505730
4: -24.6493130, 48.0615349, -8.4733324, 15.6498308, -40.2991333, 56.5202599

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0475230, upper bound: 47.0704967
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0475230, upper bound: 47.1478454
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.3680058, 42.9620438, -4.0841393, 14.7974739, -27.1654778, 46.9069595
1: -14.6970406, 49.9281464, -4.6350403, 17.1381226, -31.8351631, 54.3901787
2: -15.1878767, 48.8440247, -5.1350250, 16.6290226, -31.8168945, 53.8316422
3: -22.4419918, 52.4952850, -7.2715898, 18.0266895, -40.4686775, 59.6101646
4: -24.3620014, 47.1144180, -8.5603867, 15.8077469, -40.1697388, 55.6667404

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1041045, upper bound: 47.0886168
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1041045, upper bound: 47.1767859
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -12.6461382, 43.8475800, -4.1237211, 14.8997841, -27.5459213, 47.8315964
1: -15.0353851, 50.8989334, -4.6640100, 17.2706947, -32.3060799, 55.3880730
2: -15.4672871, 49.8316040, -5.1820674, 16.7445545, -32.2118416, 54.8645554
3: -22.8897457, 53.4193306, -7.3019671, 18.1566067, -41.0463524, 60.5621872
4: -24.6493130, 48.0615349, -8.6242256, 15.9088564, -40.5581665, 56.6760559

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0638684, upper bound: 47.1489449
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0635409, upper bound: 47.1446770
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.3680058, 42.9620438, -4.1661615, 15.0392208, -27.4072247, 46.9918785
1: -14.6970406, 49.9281464, -4.7162590, 17.4329357, -32.1299744, 54.4726067
2: -15.1878767, 48.8440247, -5.2362127, 16.9053440, -32.0932159, 53.9337845
3: -22.4419918, 52.4952850, -7.3827748, 18.3315296, -40.7735100, 59.7235870
4: -24.3620014, 47.1144180, -8.7125006, 16.0695858, -40.4315720, 55.8238640

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1629686, upper bound: 47.1764554
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0635409, upper bound: 47.1446770
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.3346949, 46.0455627, -4.1957455, 15.1547375, -28.4894295, 50.1209717
1: -15.7996721, 53.5419350, -4.7651372, 17.5528717, -33.3525429, 58.1507683
2: -16.3717403, 52.3268585, -5.2724891, 17.0362396, -33.4079819, 57.4700089
3: -24.1192474, 56.3102837, -7.4683981, 18.4647770, -42.5840225, 63.6449127
4: -26.2549019, 50.4941635, -8.7734575, 16.2064075, -42.4613113, 59.2676201

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1630569
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1630569
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -15.4945087, 52.7464104, -4.1746674, 15.0849533, -30.5794601, 56.7689438
1: -18.2942677, 61.3451691, -4.7387280, 17.4728680, -35.7671318, 65.8851624
2: -18.9236336, 59.9523926, -5.2474179, 16.9559536, -35.8795853, 65.0318146
3: -27.8473854, 64.4847412, -7.4292226, 18.3820171, -46.2294006, 71.7368317
4: -30.1493359, 57.9222450, -8.7357931, 16.1271706, -46.2765045, 66.6437302

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630569
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630569
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.3346949, 46.0455627, -4.2849174, 15.4173374, -28.7520313, 50.2126541
1: -15.7996721, 53.5419350, -4.8534427, 17.8700066, -33.6696777, 58.2404976
2: -16.3717403, 52.3268585, -5.3807850, 17.3343678, -33.7061081, 57.5796242
3: -24.1192474, 56.3102837, -7.5899143, 18.7914963, -42.9107437, 63.7687454
4: -26.2549019, 50.4941635, -8.9349260, 16.4891949, -42.7440948, 59.4290848

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603529, upper bound: 47.1677345
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603529, upper bound: 47.1677345
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -15.4945087, 52.7464104, -4.2647104, 15.3515139, -30.8460197, 56.8616333
1: -18.2942677, 61.3451691, -4.8285580, 17.7947578, -36.0890236, 65.9764099
2: -18.9236336, 59.9523926, -5.3569169, 17.2589989, -36.1826324, 65.1425934
3: -27.8473854, 64.4847412, -7.5529947, 18.7136173, -46.5610046, 71.8629227
4: -30.1493359, 57.9222450, -8.8992605, 16.4146023, -46.5639381, 66.8129501

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1615719, upper bound: 47.1677345
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1615719, upper bound: 47.1677345
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.6461382, 43.8475800, -13.3734560, 46.2076225, -58.7125931, 57.0921974
1: -15.0353851, 50.8989334, -15.8496628, 53.7283821, -68.5330734, 66.5318298
2: -15.4672871, 49.8316040, -16.4190159, 52.5117340, -67.7812576, 66.0571747
3: -22.8897457, 53.4193306, -24.1948166, 56.4987526, -79.1243515, 77.3589783
4: -24.6493130, 48.0615349, -26.3246136, 50.6681595, -75.1625290, 74.2389526

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0486395, upper bound: 47.0705274
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0486395, upper bound: 47.1369244
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.3680058, 42.9620438, -13.4117832, 46.3328934, -58.5697250, 56.2493210
1: -14.6970406, 49.9281464, -15.8963299, 53.8752975, -68.3429260, 65.6109085
2: -15.1878767, 48.8440247, -16.4676399, 52.6564484, -67.6484070, 65.1217651
3: -22.4419918, 52.4952850, -24.2661896, 56.6583176, -78.8384781, 76.5115128
4: -24.3620014, 47.1144180, -26.4058723, 50.8118477, -75.0211945, 73.3802567

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266364, upper bound: 47.0886662
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1266364, upper bound: 47.1575481
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -12.6208353, 43.7626419, -15.5239353, 52.8822975, -65.3228836, 59.1457481
1: -15.0038824, 50.8012581, -18.3305912, 61.5031471, -76.2218399, 68.9105148
2: -15.4368191, 49.7340126, -18.9601021, 60.1036453, -75.2928925, 68.4942093
3: -22.8432503, 53.3173485, -27.9018574, 64.6412735, -87.1661453, 80.9517822
4: -24.6037521, 47.9662018, -30.2063999, 58.0570297, -82.4724350, 78.0131760

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0634430, upper bound: 47.1383174
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0634430, upper bound: 47.1383174
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.3412132, 42.8723373, -15.5656719, 53.0211678, -65.1918869, 58.3018799
1: -14.6636209, 49.8248825, -18.3832092, 61.6654167, -76.0449829, 67.9901199
2: -15.1557446, 48.7409515, -19.0133343, 60.2645607, -75.1744308, 67.5581207
3: -22.3925495, 52.3874321, -27.9824181, 64.8170547, -86.8933411, 80.1077347
4: -24.3138161, 47.0137482, -30.2938747, 58.2179794, -82.3455963, 77.1554565

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1617567, upper bound: 47.1589411
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1617567, upper bound: 47.1589411
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.3346949, 46.0455627, -12.7235136, 44.1252480, -57.3425179, 58.6464653
1: -15.7996721, 53.5419350, -15.1153173, 51.2670212, -66.8688736, 68.4336472
2: -16.3717403, 52.3268585, -15.6082811, 50.1621857, -66.3559265, 67.7475433
3: -24.1192474, 56.3102837, -23.0708389, 53.8880844, -77.7713776, 79.1237259
4: -26.2549019, 50.4941635, -24.9927807, 48.4072723, -74.5315552, 75.3397903

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1520721, upper bound: 47.1552461
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1520721, upper bound: 47.1642292
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -15.4945087, 52.7464104, -12.6965485, 44.0343628, -59.3995399, 65.2838211
1: -18.2942677, 61.3451691, -15.0817461, 51.1625671, -69.2554016, 76.1543427
2: -18.9236336, 59.9523926, -15.5761366, 50.0577354, -68.7982330, 75.2968063
3: -27.8473854, 64.4847412, -23.0210171, 53.7791901, -81.3790970, 87.1994781
4: -30.1493359, 57.9222450, -24.9443798, 48.3052177, -78.3121948, 82.6897888

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0423501, upper bound: 47.0430106
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1589411, upper bound: 47.1617991
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.3346949, 46.0455627, -13.6422262, 47.0618324, -60.2893791, 59.5798950
1: -15.7996721, 53.5419350, -16.1728191, 54.7214432, -70.3302383, 69.5186996
2: -16.3717403, 52.3268585, -16.7433243, 53.4933968, -69.6963348, 68.8972244
3: -24.1192474, 56.3102837, -24.6783352, 57.5419312, -81.4434662, 80.7637100
4: -26.2549019, 50.4941635, -26.8195095, 51.6375656, -77.7754517, 77.1933441

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1607662, upper bound: 47.1604753
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1607662, upper bound: 47.1620134
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -15.4945087, 52.7464104, -13.6181641, 46.9813080, -62.3585320, 66.2250366
1: -18.2942677, 61.3451691, -16.1430454, 54.6287346, -72.7297745, 77.2497711
2: -18.9236336, 59.9523926, -16.7145805, 53.4008980, -72.1518707, 76.4552078
3: -27.8473854, 64.4847412, -24.6341267, 57.4452667, -85.0649185, 88.8504791
4: -30.1493359, 57.9222450, -26.7762661, 51.5471916, -81.5688705, 84.5522003

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1612638, upper bound: 47.1604753
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1612638, upper bound: 47.1620134
time: 0.56 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.34 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.0767665, upper bound: 47.0767665
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.0767665, upper bound: 47.1574752
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1574752, upper bound: 47.0938498
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1574752, upper bound: 47.1799104
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.0939348, upper bound: 47.1574926
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.0932470, upper bound: 47.1473520
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1652310, upper bound: 47.1768055
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1640355, upper bound: 47.1673695
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1712896
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1712896
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1712896
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1712896
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.0704967, upper bound: 47.0475230
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.0704967, upper bound: 47.1041045
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1478454, upper bound: 47.0639228
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1478454, upper bound: 47.1627924
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.0743056, upper bound: 47.0595056
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.0704967, upper bound: 47.1376392
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1603719, upper bound: 47.1558452
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1602370, upper bound: 47.1573137
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1489449, upper bound: 47.0642090
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1764554, upper bound: 47.1629686
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1446770, upper bound: 47.0635409
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1656945, upper bound: 47.1615503
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1677345, upper bound: 47.1603529
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1677345, upper bound: 47.1615719
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1677345, upper bound: 47.1603529
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1677345, upper bound: 47.1615719
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.0475230, upper bound: 47.0704967
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.0475230, upper bound: 47.1478454
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1041045, upper bound: 47.0886168
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1041045, upper bound: 47.1767859
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.0638684, upper bound: 47.1489449
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.0635409, upper bound: 47.1446770
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1629686, upper bound: 47.1764554
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.0635409, upper bound: 47.1446770
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1630569
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1630569
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630569
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630569
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1603529, upper bound: 47.1677345
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1603529, upper bound: 47.1677345
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1615719, upper bound: 47.1677345
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1615719, upper bound: 47.1677345
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.0486395, upper bound: 47.0705274
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.0486395, upper bound: 47.1369244
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1266364, upper bound: 47.0886662
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1266364, upper bound: 47.1575481
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.0634430, upper bound: 47.1383174
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.0634430, upper bound: 47.1383174
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1617567, upper bound: 47.1589411
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1617567, upper bound: 47.1589411
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1520721, upper bound: 47.1552461
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1520721, upper bound: 47.1642292
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.0423501, upper bound: 47.0430106
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1589411, upper bound: 47.1617991
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1607662, upper bound: 47.1604753
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1607662, upper bound: 47.1620134
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1612638, upper bound: 47.1604753
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.34
Output dim: 4, lower bound: -47.1612638, upper bound: 47.1620134

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.9499731, 14.3560972, -3.9499731, 14.3560972, -18.3060665, 18.3060665
1: -4.4719973, 16.6005650, -4.4719973, 16.6005650, -21.0725574, 21.0725574
2: -4.9373617, 16.1121292, -4.9373617, 16.1121292, -21.0494900, 21.0494900
3: -7.0062442, 17.3973083, -7.0062442, 17.3973083, -24.4035530, 24.4035530
4: -8.1801634, 15.2708197, -8.1801634, 15.2708197, -23.4509830, 23.4509830

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0387314, upper bound: 47.0400160
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0388421, upper bound: 47.0388421
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.9499731, 14.3560972, -4.0273848, 14.6178732, -18.5678463, 18.3834801
1: -4.4719973, 16.6005650, -4.5628810, 16.9356003, -21.4075966, 21.1634464
2: -4.9373617, 16.1121292, -5.0649624, 16.4214878, -21.3588486, 21.1770916
3: -7.0062442, 17.3973083, -7.1684437, 17.8108006, -24.8170414, 24.5657520
4: -8.1801634, 15.2708197, -8.4600706, 15.5963697, -23.7765331, 23.7308903

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0387314, upper bound: 47.1265080
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0388421, upper bound: 47.1253341
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.0273848, 14.6178732, -3.9499731, 14.3560972, -18.3834801, 18.5678463
1: -4.5628810, 16.9356003, -4.4719973, 16.6005650, -21.1634464, 21.4075947
2: -5.0649624, 16.4214878, -4.9373617, 16.1121292, -21.1770916, 21.3588486
3: -7.1684437, 17.8108006, -7.0062442, 17.3973083, -24.5657520, 24.8170433
4: -8.4600706, 15.5963697, -8.1801634, 15.2708197, -23.7308903, 23.7765331

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1359623, upper bound: 47.0684930
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1253339, upper bound: 47.0671683
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.0273848, 14.6178732, -4.0273848, 14.6178732, -18.6452579, 18.6452579
1: -4.5628810, 16.9356003, -4.5628810, 16.9356003, -21.4984818, 21.4984818
2: -5.0649624, 16.4214878, -5.0649624, 16.4214878, -21.4864502, 21.4864502
3: -7.1684437, 17.8108006, -7.1684437, 17.8108006, -24.9792404, 24.9792423
4: -8.4600706, 15.5963697, -8.4600706, 15.5963697, -24.0564404, 24.0564404

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1359625, upper bound: 47.1559525
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1253341, upper bound: 47.1536603
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.9145980, 14.2445297, -2.4636931, 9.3343706, -13.2489681, 16.7082233
1: -4.4301190, 16.4726143, -2.7771072, 10.7999125, -15.2300301, 19.2497215
2: -4.8945594, 15.9854259, -3.1412394, 10.4640884, -15.3586473, 19.1266651
3: -6.9445443, 17.2631550, -4.4422030, 11.3413839, -18.2859287, 21.7053566
4: -8.1164036, 15.1456223, -5.3976712, 9.8555918, -17.9719925, 20.5432930

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0890010, upper bound: 47.1444821
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0838805, upper bound: 47.1401904
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0937293, upper bound: 47.1563295
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0937915, upper bound: 47.1574926
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.9499731, 14.3560972, -3.8770571, 14.1431589, -18.0931320, 18.2331467
1: -4.4719973, 16.6005650, -4.3690691, 16.4045029, -20.8764992, 20.9696350
2: -4.9373617, 16.1121292, -4.8878508, 15.8842182, -20.8215790, 20.9999809
3: -7.0062442, 17.3973083, -6.8654304, 17.2504082, -24.2566528, 24.2627392
4: -8.1801634, 15.2708197, -8.1923809, 15.0538597, -23.2340183, 23.4631996

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0888276, upper bound: 47.1400930
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0839124, upper bound: 47.1349886
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0931766, upper bound: 47.1459125
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0932470, upper bound: 47.1473520
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.0273848, 14.6178732, -4.0656214, 14.6907520, -18.7181358, 18.6834946
1: -4.5628810, 16.9356003, -4.5916448, 17.0283775, -21.5912590, 21.5272427
2: -5.0649624, 16.4214878, -5.1096640, 16.4983921, -21.5633545, 21.5311508
3: -7.1684437, 17.8108006, -7.1945591, 17.9123573, -25.0808010, 25.0053558
4: -8.4600706, 15.5963697, -8.5250015, 15.6656189, -24.1256905, 24.1213722

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1640355, upper bound: 47.1673695
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1640355, upper bound: 47.1673695
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.0075064, 14.5529766, -6.1992731, 21.2122402, -25.2197456, 20.7522469
1: -4.5380826, 16.8612251, -7.0349321, 24.6053638, -29.1434460, 23.8961563
2: -5.0410957, 16.3470173, -7.6394072, 23.9174843, -28.9585800, 23.9864235
3: -7.1314745, 17.7335720, -10.8425484, 25.8447590, -32.9762344, 28.5761185
4: -8.4246225, 15.5224218, -12.3402233, 22.9567204, -31.3813419, 27.8626442

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1640355, upper bound: 47.1673695
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1640355, upper bound: 47.1673695
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.0656214, 14.6907520, -3.9600012, 14.3796635, -18.4452858, 18.6507530
1: -4.5916448, 17.0283775, -4.4894509, 16.6531467, -21.2447891, 21.5178280
2: -5.1096640, 16.4983921, -4.9817009, 16.1492329, -21.2588959, 21.4800930
3: -7.1945591, 17.9123573, -7.0482631, 17.5268631, -24.7214203, 24.9606171
4: -8.5250015, 15.6656189, -8.3357964, 15.3324518, -23.8574524, 24.0014133

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1732018, upper bound: 47.1661097
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1757743, upper bound: 47.1654130
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.0656214, 14.6907520, -6.0743117, 20.9155369, -24.9811573, 20.7650642
1: -4.5916448, 17.0283775, -6.9165897, 24.2520409, -28.8436832, 23.9449654
2: -5.1096640, 16.4983921, -7.5073729, 23.5865192, -28.6961823, 24.0057640
3: -7.1945591, 17.9123573, -10.6918325, 25.4853954, -32.6799545, 28.6041908
4: -8.5250015, 15.6656189, -12.1613083, 22.6388607, -31.1638622, 27.8269253

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1732018, upper bound: 47.1661097
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1757743, upper bound: 47.1654130
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.1992731, 21.2122402, -3.9600012, 14.3796635, -20.5789337, 25.1722393
1: -7.0349321, 24.6053638, -4.4894509, 16.6531467, -23.6880779, 29.0948143
2: -7.6394072, 23.9174843, -4.9817009, 16.1492329, -23.7886391, 28.8991852
3: -10.8425484, 25.8447590, -7.0482631, 17.5268631, -28.3694096, 32.8930168
4: -12.3402233, 22.9567204, -8.3357964, 15.3324518, -27.6726761, 31.2925148

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1652912, upper bound: 47.1649233
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1686617, upper bound: 47.1643232
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.1992731, 21.2122402, -6.0743117, 20.9155369, -27.1148052, 27.2865505
1: -7.0349321, 24.6053638, -6.9165897, 24.2520409, -31.2869720, 31.5219498
2: -7.6394072, 23.9174843, -7.5073729, 23.5865192, -31.2259254, 31.4248581
3: -10.8425484, 25.8447590, -10.6918325, 25.4853954, -36.3279419, 36.5365906
4: -12.3402233, 22.9567204, -12.1613083, 22.6388607, -34.9790840, 35.1180267

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1652912, upper bound: 47.1649233
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1652912, upper bound: 47.1643232
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.0656214, 14.6907520, -4.0656214, 14.6907520, -18.7563744, 18.7563744
1: -4.5916448, 17.0283775, -4.5916448, 17.0283775, -21.6200199, 21.6200199
2: -5.1096640, 16.4983921, -5.1096640, 16.4983921, -21.6080551, 21.6080551
3: -7.1945591, 17.9123573, -7.1945591, 17.9123573, -25.1069145, 25.1069145
4: -8.5250015, 15.6656189, -8.5250015, 15.6656189, -24.1906185, 24.1906166

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1740046, upper bound: 47.1707447
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1757743, upper bound: 47.1699473
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.0656214, 14.6907520, -6.1992731, 21.2122402, -25.2778625, 20.8900261
1: -4.5916448, 17.0283775, -7.0349321, 24.6053638, -29.1970081, 24.0633087
2: -5.1096640, 16.4983921, -7.6394072, 23.9174843, -29.0271492, 24.1377983
3: -7.1945591, 17.9123573, -10.8425484, 25.8447590, -33.0393143, 28.7549057
4: -8.5250015, 15.6656189, -12.3402233, 22.9567204, -31.4817200, 28.0058403

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1740046, upper bound: 47.1707447
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1757743, upper bound: 47.1699473
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.1992731, 21.2122402, -4.0656214, 14.6907520, -20.8900261, 25.2778625
1: -7.0349321, 24.6053638, -4.5916448, 17.0283775, -24.0633087, 29.1970062
2: -7.6394072, 23.9174843, -5.1096640, 16.4983921, -24.1377983, 29.0271492
3: -10.8425484, 25.8447590, -7.1945591, 17.9123573, -28.7549057, 33.0393143
4: -12.3402233, 22.9567204, -8.5250015, 15.6656189, -28.0058403, 31.4817200

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1665553, upper bound: 47.1679135
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1670535, upper bound: 47.1678911
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.1992731, 21.2122402, -6.1992731, 21.2122402, -27.4115124, 27.4115124
1: -7.0349321, 24.6053638, -7.0349321, 24.6053638, -31.6402969, 31.6402912
2: -7.6394072, 23.9174843, -7.6394072, 23.9174843, -31.5568924, 31.5568924
3: -10.8425484, 25.8447590, -10.8425484, 25.8447590, -36.6873055, 36.6873055
4: -12.3402233, 22.9567204, -12.3402233, 22.9567204, -35.2969437, 35.2969398

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1665553, upper bound: 47.1679135
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1670535, upper bound: 47.1678911
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.8461647, 14.0256577, -12.6461382, 43.8475800, -47.5509033, 26.6717949
1: -4.3524222, 16.2163048, -15.0353851, 50.8989334, -55.0746880, 31.2516899
2: -4.8096867, 15.7374773, -15.4672871, 49.8316040, -54.4933891, 31.2047653
3: -6.8268933, 16.9921341, -22.8897457, 53.4193306, -60.0871468, 39.8818817
4: -7.9824967, 14.9052515, -24.6493130, 48.0615349, -56.0281677, 39.5545654

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0699305, upper bound: 47.0474119
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0661130, upper bound: 47.0451142
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0704965, upper bound: 47.0461146
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0704967, upper bound: 47.0475230
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.8461647, 14.0256577, -12.3680058, 42.9620438, -46.6662178, 26.3936634
1: -4.3524222, 16.2163048, -14.6970406, 49.9281464, -54.1025162, 30.9133453
2: -4.8096867, 15.7374773, -15.1878767, 48.8440247, -53.5037308, 30.9253540
3: -6.8268933, 16.9921341, -22.4419918, 52.4952850, -59.1607056, 39.4341240
4: -7.9824967, 14.9052515, -24.3620014, 47.1144180, -55.0795174, 39.2672501

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0699305, upper bound: 47.1038839
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0661130, upper bound: 47.1006637
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0704965, upper bound: 47.1037375
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0704967, upper bound: 47.1036697
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -3.9194047, 14.2739916, -12.6461382, 43.8475800, -47.6227951, 26.9201298
1: -4.4381156, 16.5363541, -15.0353851, 50.8989334, -55.1592827, 31.5717373
2: -4.9312482, 16.0302601, -15.4672871, 49.8316040, -54.6123695, 31.4975471
3: -6.9790220, 17.3885593, -22.8897457, 53.4193306, -60.2367706, 40.2783051
4: -8.2539015, 15.2128563, -24.6493130, 48.0615349, -56.2986603, 39.8621674

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1478453, upper bound: 47.0638684
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1391912, upper bound: 47.0622322
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -3.9194047, 14.2739916, -12.3680058, 42.9620438, -46.7406960, 26.6419964
1: -4.4381156, 16.5363541, -14.6970406, 49.9281464, -54.1921310, 31.2333946
2: -4.9312482, 16.0302601, -15.1878767, 48.8440247, -53.6283226, 31.2181339
3: -6.9790220, 17.3885593, -22.4419918, 52.4952850, -59.3183060, 39.8305511
4: -8.2539015, 15.2128563, -24.3620014, 47.1144180, -55.3602867, 39.5748520

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1478454, upper bound: 47.1626641
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1391913, upper bound: 47.1582495
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.9499731, 14.3560972, -13.6301708, 47.0246315, -50.8509178, 27.9862652
1: -4.4719973, 16.6005650, -16.1748352, 54.6406326, -58.9514618, 32.7753983
2: -4.9373617, 16.1121292, -16.6790943, 53.4380875, -58.2469673, 32.7912216
3: -7.0062442, 17.3973083, -24.6280060, 57.3694382, -64.2423019, 42.0253143
4: -8.1801634, 15.2708197, -26.6065903, 51.5566139, -59.7367783, 41.8774109

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0733058, upper bound: 47.0579059
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0661130, upper bound: 47.0557951
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0616800, upper bound: 47.0502923
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0743054, upper bound: 47.0580972
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0743056, upper bound: 47.0595056
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.9499731, 14.3560972, -13.2847500, 45.8887329, -49.7139893, 27.6408405
1: -4.4719973, 16.6005650, -15.7483788, 53.3632355, -57.6707230, 32.3489456
2: -4.9373617, 16.1121292, -16.3181915, 52.1585617, -56.9602318, 32.4303207
3: -7.0062442, 17.3973083, -24.0418644, 56.1395454, -63.0031586, 41.4391708
4: -8.1801634, 15.2708197, -26.1825333, 50.3334961, -58.5133629, 41.4533539

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0733058, upper bound: 47.1241650
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0699264, upper bound: 47.1329671
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0639867, upper bound: 47.1290429
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0704965, upper bound: 47.1364816
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0743056, upper bound: 47.1376392
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.0273848, 14.6178732, -13.3346949, 46.0455627, -49.9510994, 27.9525681
1: -4.5628810, 16.9356003, -15.7996721, 53.5419350, -57.9474144, 32.7352715
2: -5.0649624, 16.4214878, -16.3717403, 52.3268585, -57.2630844, 32.7932243
3: -7.1684437, 17.8108006, -24.1192474, 56.3102837, -63.3457489, 41.9300461
4: -8.4600706, 15.5963697, -26.2549019, 50.4941635, -58.9542313, 41.8512650

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1602370, upper bound: 47.1558452
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1602370, upper bound: 47.1558452
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.0075064, 14.5529766, -15.4945087, 52.7464104, -56.6003456, 30.0474834
1: -4.5380826, 16.8612251, -18.2942677, 61.3451691, -65.6834412, 35.1554794
2: -5.0410957, 16.3470173, -18.9236336, 59.9523926, -64.8260651, 35.2706451
3: -7.1314745, 17.7335720, -27.8473854, 64.4847412, -71.4398727, 45.5809479
4: -8.4246225, 15.5224218, -30.1493359, 57.9222450, -66.3327255, 45.6717567

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1602370, upper bound: 47.1573137
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1602370, upper bound: 47.1573137
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.9160135, 14.2161093, -12.6461382, 43.8475800, -47.6236801, 26.8622475
1: -4.4194160, 16.4785137, -15.0353851, 50.8989334, -55.1446152, 31.5138988
2: -4.9253159, 15.9602757, -15.4672871, 49.8316040, -54.6091042, 31.4275627
3: -6.9331017, 17.3285389, -22.8897457, 53.4193306, -60.1949577, 40.2182846
4: -8.2360659, 15.1366425, -24.6493130, 48.0615349, -56.2903442, 39.7859459

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1218253, upper bound: 47.0593404
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1444123, upper bound: 47.0617563
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.9491742, 14.3233261, -12.3680058, 42.9620438, -46.7745667, 26.6913242
1: -4.4580026, 16.6038208, -14.6970406, 49.9281464, -54.2154884, 31.3008595
2: -4.9676933, 16.0823917, -15.1878767, 48.8440247, -53.6666260, 31.2702675
3: -6.9926357, 17.4643440, -22.4419918, 52.4952850, -59.3351364, 39.9063339
4: -8.3071480, 15.2576637, -24.3620014, 47.1144180, -55.4209633, 39.6196632

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1710523, upper bound: 47.1612743
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1725572, upper bound: 47.1593699
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1685078, upper bound: 47.1586492
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.0401611, 20.7042751, -12.6208353, 43.7626419, -49.6493988, 33.3251038
1: -6.8483753, 24.0140572, -15.0038824, 50.8012581, -57.4573402, 39.0179367
2: -7.4441037, 23.3368034, -15.4368191, 49.7340126, -57.0299988, 38.7736130
3: -10.5598764, 25.2176781, -22.8432503, 53.3173485, -63.7108345, 48.0609283
4: -12.0327148, 22.3857880, -24.6037521, 47.9662018, -59.9745140, 46.9895401

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1441049, upper bound: 47.0635409
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1441049, upper bound: 47.0635409
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.0817451, 20.8432217, -12.3412132, 42.8723373, -48.8048973, 33.1844330
1: -6.8995233, 24.1751728, -14.6636209, 49.8248825, -56.5361824, 38.8387947
2: -7.4971895, 23.4976387, -15.1557446, 48.7409515, -56.0932846, 38.6533813
3: -10.6402817, 25.3906555, -22.3925495, 52.3874321, -62.8662491, 47.7832031
4: -12.1190033, 22.5459404, -24.3138161, 47.0137482, -59.1151314, 46.8597565

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1651224, upper bound: 47.1615503
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1651224, upper bound: 47.1615503
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.0656214, 14.6907520, -13.3346949, 46.0455627, -49.9928207, 28.0254478
1: -4.5916448, 17.0283775, -15.7996721, 53.5419350, -57.9800606, 32.8280487
2: -5.1096640, 16.4983921, -16.3717403, 52.3268585, -57.3101768, 32.8701324
3: -7.1945591, 17.9123573, -24.1192474, 56.3102837, -63.3753052, 42.0316048
4: -8.5250015, 15.6656189, -26.2549019, 50.4941635, -59.0191574, 41.9205208

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1717688, upper bound: 47.1589038
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1747594, upper bound: 47.1582046
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.0656214, 14.6907520, -15.4945087, 52.7464104, -56.6620331, 30.1852608
1: -4.5916448, 17.0283775, -18.2942677, 61.3451691, -65.7409668, 35.3226357
2: -5.1096640, 16.4983921, -18.9236336, 59.9523926, -64.8969193, 35.4220276
3: -7.1945591, 17.9123573, -27.8473854, 64.4847412, -71.5063324, 45.7597351
4: -8.5250015, 15.6656189, -30.1493359, 57.9222450, -66.4414139, 45.8149567

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1717688, upper bound: 47.1608437
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1747594, upper bound: 47.1601541
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.1992731, 21.2122402, -13.3346949, 46.0455627, -52.1136818, 34.5469360
1: -7.0349321, 24.6053638, -15.7996721, 53.5419350, -60.4046135, 40.4050331
2: -7.6394072, 23.9174843, -16.3717403, 52.3268585, -59.8404884, 40.2892227
3: -10.8425484, 25.8447590, -24.1192474, 56.3102837, -67.0151062, 49.9640045
4: -12.3402233, 22.9567204, -26.2549019, 50.4941635, -62.8343811, 49.2116241

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1624071, upper bound: 47.1449880
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1608933, upper bound: 47.1574191
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1651124, upper bound: 47.1568614
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.1992731, 21.2122402, -15.4945087, 52.7464104, -58.7828941, 36.7067490
1: -7.0349321, 24.6053638, -18.2942677, 61.3451691, -68.1655197, 42.8996277
2: -7.6394072, 23.9174843, -18.9236336, 59.9523926, -67.4272308, 42.8411179
3: -10.8425484, 25.8447590, -27.8473854, 64.4847412, -75.1461258, 53.6921425
4: -12.3402233, 22.9567204, -30.1493359, 57.9222450, -70.2392578, 53.1060562

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1670847, upper bound: 47.1449880
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 31

Time for candidate selection: 5.24 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1328950, upper bound: 47.1043297
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1563035, upper bound: 47.1241471
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1663010, upper bound: 47.1378153
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1361816, upper bound: 47.1358456
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -12.6461382, 43.8475800, -3.8461647, 14.0256577, -26.6717949, 47.5509033
1: -15.0353851, 50.8989334, -4.3524222, 16.2163048, -31.2516899, 55.0746880
2: -15.4672871, 49.8316040, -4.8096867, 15.7374773, -31.2047653, 54.4933891
3: -22.8897457, 53.4193306, -6.8268933, 16.9921341, -39.8818817, 60.0871429
4: -24.6493130, 48.0615349, -7.9824967, 14.9052515, -39.5545654, 56.0281677

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0231285, upper bound: 47.0414190
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0475230, upper bound: 47.0696372
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0475230, upper bound: 47.0704967
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.6461382, 43.8475800, -3.9194047, 14.2739916, -26.9201298, 47.6227951
1: -15.0353851, 50.8989334, -4.4381156, 16.5363541, -31.5717392, 55.1592827
2: -15.4672871, 49.8316040, -4.9312482, 16.0302601, -31.4975452, 54.6123695
3: -22.8897457, 53.4193306, -6.9790220, 17.3885593, -40.2783051, 60.2367744
4: -24.6493130, 48.0615349, -8.2539015, 15.2128563, -39.8621674, 56.2986603

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0231285, upper bound: 47.0575695
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0475230, upper bound: 47.1470479
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0475230, upper bound: 47.1478454
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.3680058, 42.9620438, -3.8461647, 14.0256577, -26.3936634, 46.6662178
1: -14.6970406, 49.9281464, -4.3524222, 16.2163048, -30.9133453, 54.1025162
2: -15.1878767, 48.8440247, -4.8096867, 15.7374773, -30.9253540, 53.5037308
3: -22.4419918, 52.4952850, -6.8268933, 16.9921341, -39.4341240, 59.1607094
4: -24.3620014, 47.1144180, -7.9824967, 14.9052515, -39.2672501, 55.0795135

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0765287, upper bound: 47.0608190
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1034106, upper bound: 47.0874042
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1034106, upper bound: 47.0886168
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.3680058, 42.9620438, -3.9194047, 14.2739916, -26.6419964, 46.7406960
1: -14.6970406, 49.9281464, -4.4381156, 16.5363541, -31.2333946, 54.1921310
2: -15.1878767, 48.8440247, -4.9312482, 16.0302601, -31.2181339, 53.6283226
3: -22.4419918, 52.4952850, -6.9790220, 17.3885593, -39.8305511, 59.3183060
4: -24.3620014, 47.1144180, -8.2539015, 15.2128563, -39.5748520, 55.3602867

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0765287, upper bound: 47.1406204
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1034106, upper bound: 47.1596587
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1034106, upper bound: 47.1596587
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -12.6461382, 43.8475800, -3.9160135, 14.2161093, -26.8622475, 47.6236801
1: -15.0353851, 50.8989334, -4.4194160, 16.4785137, -31.5138988, 55.1446152
2: -15.4672871, 49.8316040, -4.9253159, 15.9602757, -31.4275627, 54.6091042
3: -22.8897457, 53.4193306, -6.9331017, 17.3285389, -40.2182846, 60.1949539
4: -24.6493130, 48.0615349, -8.2360659, 15.1366425, -39.7859459, 56.2903442

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0638684, upper bound: 47.1481384
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0638684, upper bound: 47.1489449
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.6208353, 43.7626419, -6.0401611, 20.7042751, -33.3250999, 49.6493988
1: -15.0038824, 50.8012581, -6.8483753, 24.0140572, -39.0179367, 57.4573441
2: -15.4368191, 49.7340126, -7.4441037, 23.3368034, -38.7736130, 57.0299988
3: -22.8432503, 53.3173485, -10.5598764, 25.2176781, -48.0609283, 63.7108383
4: -24.6037521, 47.9662018, -12.0327148, 22.3857880, -46.9895401, 59.9745140

Time for backsubstitution: 1.93 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1666667, mid=0.1666667, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 1) starts
Candidate diff: 0.0833333


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

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
- Time for IS candidates: 1.14 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.14
Output dim: 4, lower bound: -47.1752727, upper bound: 47.1684808
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.14
Output dim: 4, lower bound: -47.1679581, upper bound: 47.1679581

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.3289332, 15.6493454, -7.0806756, 24.7960377, -29.1249695, 22.7300167
1: -4.9227071, 18.1375618, -8.1733112, 28.7153568, -33.6380653, 26.3108730
2: -5.4355350, 17.6055393, -8.7818174, 28.0788403, -33.5143738, 26.3873558
3: -7.7107296, 19.0719624, -12.6078606, 30.1267490, -37.8374786, 31.6798229
4: -9.0420084, 16.7446327, -14.1291494, 27.0256519, -36.0676460, 30.8737793

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

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
time: 0.51 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -13.8669882, 47.8264809, -8.6101542, 29.6928444, -43.5598297, 56.3413658
1: -16.4424095, 55.6123199, -9.9591999, 34.3566132, -50.7990112, 65.4222336
2: -17.0157623, 54.3669052, -10.5797205, 33.6746674, -50.6904259, 64.8336792
3: -25.0840302, 58.4735527, -15.2551098, 36.0304794, -61.1145096, 73.5841751
4: -27.2452755, 52.4822922, -16.8258591, 32.5553856, -59.8006592, 69.3081512

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679581, upper bound: 47.1679581
time: 0.79 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679581, upper bound: 47.1679581
time: 0.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.13 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 4, lower bound: -47.1679581, upper bound: 47.1679581
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 4, lower bound: -47.1679581, upper bound: 47.1679581
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 4, lower bound: -47.1679581, upper bound: 47.1679581
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 4, lower bound: -47.1679581, upper bound: 47.1679581

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -4.3289332, 15.6493454, -4.3289332, 15.6493454, -19.9782753, 19.9782753
1: -4.9227071, 18.1375618, -4.9227071, 18.1375618, -23.0602684, 23.0602684
2: -5.4355350, 17.6055393, -5.4355350, 17.6055393, -23.0410748, 23.0410748
3: -7.7107296, 19.0719624, -7.7107296, 19.0719624, -26.7826920, 26.7826920
4: -9.0420084, 16.7446327, -9.0420084, 16.7446327, -25.7866364, 25.7866364

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1746476, upper bound: 47.1680955
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1752691, upper bound: 47.1684535
time: 0.50 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -4.3289332, 15.6493454, -12.8257484, 44.2895927, -48.4904671, 28.4750900
1: -4.9227071, 18.1375618, -15.1263590, 51.5547981, -56.3153458, 33.2639198
2: -5.4355350, 17.6055393, -15.7967310, 50.3415146, -55.6429100, 33.4022713
3: -7.7107296, 19.0719624, -23.2090626, 54.2801094, -61.8550987, 42.2810173
4: -9.0420084, 16.7446327, -25.4259567, 48.5490341, -57.5910339, 42.1705894

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1746476, upper bound: 47.1680955
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1752691, upper bound: 47.1684535
time: 0.56 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -13.8669882, 47.8264809, -4.3289332, 15.6493454, -29.5163288, 52.0300331
1: -16.4424095, 55.6123199, -4.9227071, 18.1375618, -34.5799713, 60.3761864
2: -17.0157623, 54.3669052, -5.4355350, 17.6055393, -34.6212921, 59.6714478
3: -25.0840302, 58.4735527, -7.7107296, 19.0719624, -44.1559906, 66.0495911
4: -27.2452755, 52.4822922, -9.0420084, 16.7446327, -43.9899063, 61.5242958

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1646270, upper bound: 47.1619679
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622281, upper bound: 47.1622281
time: 0.82 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -13.8669882, 47.8264809, -13.0686092, 45.0560722, -58.8123169, 60.7855453
1: -16.4424095, 55.6123199, -15.4047651, 52.4332733, -68.6767883, 70.8160324
2: -17.0157623, 54.3669052, -16.0822449, 51.1983566, -68.0413132, 70.2737198
3: -25.0840302, 58.4735527, -23.6169319, 55.1978226, -80.0572128, 81.8583908
4: -27.2452755, 52.4822922, -25.8539028, 49.3803787, -76.5030136, 78.2091293

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1646270, upper bound: 47.1619679
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622281, upper bound: 47.1622281
time: 0.92 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.57 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 4, lower bound: -47.1746476, upper bound: 47.1680955
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 4, lower bound: -47.1752691, upper bound: 47.1684535
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 4, lower bound: -47.1746476, upper bound: 47.1680955
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 4, lower bound: -47.1752691, upper bound: 47.1684535
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 4, lower bound: -47.1646270, upper bound: 47.1619679
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 4, lower bound: -47.1622281, upper bound: 47.1622281
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 4, lower bound: -47.1646270, upper bound: 47.1619679
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 4, lower bound: -47.1622281, upper bound: 47.1622281

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.1957455, 15.1547375, -4.3289332, 15.6493454, -19.8450909, 19.4836674
1: -4.7651372, 17.5528717, -4.9227071, 18.1375618, -22.9026985, 22.4755783
2: -5.2724891, 17.0362396, -5.4355350, 17.6055393, -22.8780251, 22.4717751
3: -7.4683981, 18.4647770, -7.7107296, 19.0719624, -26.5403595, 26.1755066
4: -8.7734575, 16.2064075, -9.0420084, 16.7446327, -25.5180874, 25.2484131

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1801766, upper bound: 47.1801766
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1801766, upper bound: 47.1803884
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.2849174, 15.4173374, -4.2925625, 15.5288219, -19.8137379, 19.7098999
1: -4.8534427, 17.8700066, -4.8777080, 17.9997559, -22.8531971, 22.7477093
2: -5.3807850, 17.3343678, -5.3916469, 17.4679737, -22.8487587, 22.7260151
3: -7.5899143, 18.7914963, -7.6434240, 18.9274788, -26.5173931, 26.4349174
4: -8.9349260, 16.4891949, -8.9751225, 16.6096172, -25.5445385, 25.4643154

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1803884, upper bound: 47.1804113
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1803884, upper bound: 47.1806231
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.1957455, 15.1547375, -12.8257484, 44.2895927, -48.3568802, 27.9804802
1: -4.7651372, 17.5528717, -15.1263590, 51.5547981, -56.1573334, 32.6792297
2: -5.2724891, 17.0362396, -15.7967310, 50.3415146, -55.4796333, 32.8329697
3: -7.4683981, 18.4647770, -23.2090626, 54.2801094, -61.6117668, 41.6738396
4: -8.7734575, 16.2064075, -25.4259567, 48.5490341, -57.3224869, 41.6323624

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1669999, upper bound: 47.1636863
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1668927, upper bound: 47.1667574
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.2849174, 15.4173374, -12.7795544, 44.1365738, -48.2960777, 28.1968880
1: -4.8534427, 17.8700066, -15.0706711, 51.3774223, -56.0701675, 32.9406776
2: -5.3807850, 17.3343678, -15.7413015, 50.1662140, -55.4143867, 33.0756683
3: -7.5899143, 18.7914963, -23.1258087, 54.0946503, -61.5506821, 41.9173050
4: -8.9349260, 16.4891949, -25.3413944, 48.3782158, -57.3131371, 41.8305893

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1713870, upper bound: 47.1647151
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712801, upper bound: 47.1677862
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.7235136, 44.1252480, -3.9300065, 14.3691902, -27.0927010, 47.9186287
1: -15.1153173, 51.2670212, -4.4566350, 16.6566200, -31.7719383, 55.5586891
2: -15.6082811, 50.1621857, -4.9421859, 16.1524506, -31.7607307, 54.9636650
3: -23.0708389, 53.8880844, -7.0054507, 17.5057449, -40.5765839, 60.7388115
4: -24.9927807, 48.4072723, -8.2796173, 15.3206472, -40.3134270, 56.6735954

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1646811, upper bound: 47.1723357
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1650724, upper bound: 47.1731129
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.6422262, 47.0618324, -4.3256726, 15.6378422, -29.2800674, 51.2719955
1: -16.1728191, 54.7214432, -4.9189992, 18.1239510, -34.2967682, 59.4915009
2: -16.7433243, 53.4933968, -5.4315004, 17.5925045, -34.3358231, 58.8029060
3: -24.6783352, 57.5419312, -7.7049551, 19.0577106, -43.7360458, 65.1221695
4: -26.8195095, 51.6375656, -9.0353460, 16.7323036, -43.5518074, 60.6729126

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622874, upper bound: 47.1734464
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1625165, upper bound: 47.1740268
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.7235136, 44.1252480, -12.5608091, 43.4088554, -55.9988747, 56.5793915
1: -15.1153173, 51.2670212, -14.8104725, 50.5115204, -65.3975067, 65.8779984
2: -15.6082811, 50.1621857, -15.4711943, 49.3233070, -64.7394562, 65.4586182
3: -23.0708389, 53.8880844, -22.7202358, 53.1783791, -75.9905319, 76.3630600
4: -24.9927807, 48.4072723, -24.8940926, 47.5639153, -72.4078217, 73.1619720

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1646136, upper bound: 47.1607066
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1644444, upper bound: 47.1619550
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.6422262, 47.0618324, -13.0638866, 45.0398750, -58.5706482, 60.0259743
1: -16.1728191, 54.7214432, -15.3991804, 52.4143105, -68.3901367, 69.9294128
2: -16.7433243, 53.4933968, -16.0764732, 51.1798515, -67.7500839, 69.4033432
3: -24.6783352, 57.5419312, -23.6083984, 55.1779976, -79.6357956, 80.9281540
4: -26.8195095, 51.6375656, -25.8448219, 49.3625183, -76.0628510, 77.3618393

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622151, upper bound: 47.1607655
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1620132, upper bound: 47.1620132
time: 0.55 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.90 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 4, lower bound: -47.1801766, upper bound: 47.1801766
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 4, lower bound: -47.1801766, upper bound: 47.1803884
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 4, lower bound: -47.1803884, upper bound: 47.1804113
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 4, lower bound: -47.1803884, upper bound: 47.1806231
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 4, lower bound: -47.1669999, upper bound: 47.1636863
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 4, lower bound: -47.1668927, upper bound: 47.1667574
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 4, lower bound: -47.1713870, upper bound: 47.1647151
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 4, lower bound: -47.1712801, upper bound: 47.1677862
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 4, lower bound: -47.1646811, upper bound: 47.1723357
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 4, lower bound: -47.1650724, upper bound: 47.1731129
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 4, lower bound: -47.1622874, upper bound: 47.1734464
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 4, lower bound: -47.1625165, upper bound: 47.1740268
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 4, lower bound: -47.1646136, upper bound: 47.1607066
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 4, lower bound: -47.1644444, upper bound: 47.1619550
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 4, lower bound: -47.1622151, upper bound: 47.1607655
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 4, lower bound: -47.1620132, upper bound: 47.1620132

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.1957455, 15.1547375, -4.1957455, 15.1547375, -19.3504810, 19.3504829
1: -4.7651372, 17.5528717, -4.7651372, 17.5528717, -22.3180084, 22.3180084
2: -5.2724891, 17.0362396, -5.2724891, 17.0362396, -22.3087254, 22.3087254
3: -7.4683981, 18.4647770, -7.4683981, 18.4647770, -25.9331741, 25.9331741
4: -8.7734575, 16.2064075, -8.7734575, 16.2064075, -24.9798641, 24.9798641

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1730214, upper bound: 47.1674367
time: 0.48 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1658737
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.1957455, 15.1547375, -4.2849174, 15.4173374, -19.6130829, 19.4396477
1: -4.7651372, 17.5528717, -4.8534427, 17.8700066, -22.6351433, 22.4063129
2: -5.2724891, 17.0362396, -5.3807850, 17.3343678, -22.6068497, 22.4170246
3: -7.4683981, 18.4647770, -7.5899143, 18.7914963, -26.2598953, 26.0546913
4: -8.7734575, 16.2064075, -8.9349260, 16.4891949, -25.2626514, 25.1413326

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1730214, upper bound: 47.1718533
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1702607
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.2849174, 15.4173374, -4.1957455, 15.1547375, -19.4396477, 19.6130829
1: -4.8534427, 17.8700066, -4.7651372, 17.5528717, -22.4063148, 22.6351433
2: -5.3807850, 17.3343678, -5.2724891, 17.0362396, -22.4170246, 22.6068516
3: -7.5899143, 18.7914963, -7.4683981, 18.4647770, -26.0546913, 26.2598953
4: -8.9349260, 16.4891949, -8.7734575, 16.2064075, -25.1413326, 25.2626514

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1735555, upper bound: 47.1676564
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.2849174, 15.4173374, -4.2849174, 15.4173374, -19.7022495, 19.7022495
1: -4.8534427, 17.8700066, -4.8534427, 17.8700066, -22.7234478, 22.7234478
2: -5.3807850, 17.3343678, -5.3807850, 17.3343678, -22.7151527, 22.7151527
3: -7.5899143, 18.7914963, -7.5899143, 18.7914963, -26.3814106, 26.3814106
4: -8.9349260, 16.4891949, -8.9349260, 16.4891949, -25.4241199, 25.4241199

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1735555, upper bound: 47.1720844
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1712896
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.1611032, 15.0402527, -12.5386314, 43.3358650, -47.3638229, 27.5788841
1: -4.7245216, 17.4201145, -14.7782555, 50.4466515, -55.0019684, 32.1983681
2: -5.2298994, 16.9050446, -15.4454908, 49.2435837, -54.3325272, 32.3505325
3: -7.4067106, 18.3266754, -22.6835632, 53.1199379, -60.3825684, 41.0102386
4: -8.7091722, 16.0773907, -24.8940659, 47.4709549, -56.1801262, 40.9714546

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1668927, upper bound: 47.1636863
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1668927, upper bound: 47.1636863
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.1008406, 14.8422709, -14.5672054, 49.6573448, -53.5917549, 29.4094734
1: -4.6457920, 17.1946945, -17.1258297, 57.8160286, -62.2493134, 34.3205223
2: -5.1589270, 16.6772480, -17.8585300, 56.4359894, -61.4137268, 34.5357780
3: -7.2909489, 18.0929794, -26.1972809, 60.8432045, -67.9440308, 44.2902603
4: -8.6035089, 15.8505039, -28.5813999, 54.4767380, -63.0597076, 44.4318924

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1668927, upper bound: 47.1667574
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1668927, upper bound: 47.1667574
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.2527108, 15.3104954, -12.4907398, 43.1770287, -47.2996178, 27.8012314
1: -4.8148193, 17.7463245, -14.7196579, 50.2626801, -54.9105110, 32.4659805
2: -5.3409400, 17.2112198, -15.3879776, 49.0610275, -54.2626801, 32.5991936
3: -7.5317721, 18.6622658, -22.5956783, 52.9271393, -60.3176880, 41.2579422
4: -8.8748684, 16.3677406, -24.8065910, 47.2923355, -56.1672058, 41.1743317

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712801, upper bound: 47.1647151
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712801, upper bound: 47.1647151
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.1864662, 15.0959291, -14.5215263, 49.5060539, -53.5292053, 29.6174526
1: -4.7318096, 17.5026073, -17.0701180, 57.6409378, -62.1622696, 34.5727196
2: -5.2641234, 16.9664192, -17.8036995, 56.2622681, -61.3469505, 34.7701187
3: -7.4095511, 18.4113407, -26.1135445, 60.6597366, -67.8821487, 44.5248795
4: -8.7605743, 16.1251717, -28.4977646, 54.3065796, -63.0528336, 44.6229362

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1653239, upper bound: 47.1658022
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1686943, upper bound: 47.1652021
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -12.7235136, 44.1252480, -3.7945487, 13.8754206, -26.5989342, 47.7827721
1: -15.1153173, 51.2670212, -4.2991891, 16.0663052, -31.1816216, 55.4009132
2: -15.6082811, 50.1621857, -4.7751355, 15.5799980, -31.1882782, 54.7965164
3: -23.0708389, 53.8880844, -6.7621970, 16.8902683, -39.9611053, 60.4943810
4: -24.9927807, 48.4072723, -8.0042782, 14.7790022, -39.7717819, 56.3956032

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1561502, upper bound: 47.1697346
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1561502, upper bound: 47.1723357
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.6643715, 43.9309196, -3.8281255, 13.9594917, -26.6238556, 47.6247940
1: -15.0440998, 51.0419197, -4.3222551, 16.1841278, -31.2282276, 55.1998177
2: -15.5382872, 49.9401588, -4.8204536, 15.6792259, -31.2175121, 54.6206589
3: -22.9651928, 53.6530914, -6.7869787, 17.0132713, -39.9784622, 60.2867851
4: -24.8865948, 48.1910629, -8.0733624, 14.8673763, -39.7539711, 56.2532539

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1563946, upper bound: 47.1706184
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1563946, upper bound: 47.1731129
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.6422262, 47.0618324, -4.1924667, 15.1432247, -28.7854500, 51.1383934
1: -16.1728191, 54.7214432, -4.7614188, 17.5392609, -33.7120781, 59.3334999
2: -16.7433243, 53.4933968, -5.2684364, 17.0231972, -33.7665100, 58.6396103
3: -24.6783352, 57.5419312, -7.4626141, 18.4505215, -43.1288567, 64.8788300
4: -26.8195095, 51.6375656, -8.7667761, 16.1940765, -43.0135803, 60.4043427

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1632543
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630553
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.5861940, 46.8778954, -4.2823873, 15.4078484, -28.9940395, 51.0471687
1: -16.1056728, 54.5083275, -4.8504467, 17.8587418, -33.9644165, 59.2111015
2: -16.6765842, 53.2831993, -5.3774796, 17.3235989, -34.0001793, 58.5399742
3: -24.5787296, 57.3194237, -7.5852194, 18.7796097, -43.3583374, 64.7815170
4: -26.7181225, 51.4329720, -8.9293203, 16.4790897, -43.1972122, 60.3622932

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603529, upper bound: 47.1679002
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1615719, upper bound: 47.1677334
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -12.6771326, 43.9738846, -12.2439728, 42.3562660, -54.8914948, 56.1057701
1: -15.0596056, 51.0911331, -14.4265337, 49.2907829, -64.1118774, 65.3146820
2: -15.5524607, 49.9890289, -15.0843668, 48.1135788, -63.4662056, 64.8940811
3: -22.9876213, 53.7042656, -22.1421833, 51.9016037, -74.6218109, 75.5990448
4: -24.9081554, 48.2376060, -24.3096695, 46.3754387, -71.1298141, 72.4045715

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0629257, upper bound: 47.1213464
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0629257, upper bound: 47.1575457
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.5913439, 43.6804390, -14.3714476, 48.9743423, -61.3780785, 57.9263115
1: -14.9509096, 50.7557068, -16.8807278, 56.9925308, -71.6377487, 67.4305649
2: -15.4507589, 49.6512146, -17.6058102, 55.6339989, -70.8237000, 67.0702133
3: -22.8267250, 53.3548393, -25.8161678, 59.9731293, -82.4669342, 78.9113464
4: -24.7555275, 47.9082298, -28.1660480, 53.6959877, -78.2567291, 75.9191055

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0629325, upper bound: 47.1258307
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0629325, upper bound: 47.1589403
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.5964956, 46.9106598, -12.7497654, 43.9991837, -57.4786263, 59.5554886
1: -16.1172180, 54.5460396, -15.0207338, 51.2055168, -67.1167755, 69.3726501
2: -16.6880341, 53.3198357, -15.6935768, 49.9844093, -66.4915314, 68.8425751
3: -24.5951366, 57.3586617, -23.0376320, 53.9141006, -78.2802963, 80.1720810
4: -26.7354984, 51.4671974, -25.2652245, 48.1889725, -74.8013306, 76.6088104

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1607655, upper bound: 47.1607655
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1607655, upper bound: 47.1607655
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.5240974, 46.6667824, -14.8889523, 50.6483345, -64.0230255, 61.4406357
1: -16.0267487, 54.2665520, -17.4876347, 58.9481316, -74.7255402, 71.5605011
2: -16.6022205, 53.0395737, -18.2277164, 57.5445175, -73.9244843, 71.0927200
3: -24.4613953, 57.0677147, -26.7319660, 62.0291901, -86.2148056, 83.5656662
4: -26.6071854, 51.1942558, -29.1401997, 55.5493774, -82.0052109, 80.1996536

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1607655, upper bound: 47.1620132
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1607655, upper bound: 47.1620132
time: 0.60 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.39 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1730214, upper bound: 47.1674367
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1658737
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1730214, upper bound: 47.1718533
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1702607
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1735555, upper bound: 47.1676564
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1735555, upper bound: 47.1720844
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1712896
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1668927, upper bound: 47.1636863
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1668927, upper bound: 47.1636863
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1668927, upper bound: 47.1667574
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1668927, upper bound: 47.1667574
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1712801, upper bound: 47.1647151
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1712801, upper bound: 47.1647151
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1653239, upper bound: 47.1658022
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1686943, upper bound: 47.1652021
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1561502, upper bound: 47.1697346
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1561502, upper bound: 47.1723357
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1563946, upper bound: 47.1706184
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1563946, upper bound: 47.1731129
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1632543
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630553
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1603529, upper bound: 47.1679002
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1615719, upper bound: 47.1677334
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.0629257, upper bound: 47.1213464
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.0629257, upper bound: 47.1575457
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.0629325, upper bound: 47.1258307
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.0629325, upper bound: 47.1589403
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1607655, upper bound: 47.1607655
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1607655, upper bound: 47.1607655
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1607655, upper bound: 47.1620132
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 4, lower bound: -47.1607655, upper bound: 47.1620132

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.9600012, 14.3796635, -4.1611032, 15.0402527, -19.0002537, 18.5407677
1: -4.4894509, 16.6531467, -4.7245216, 17.4201145, -21.9095650, 21.3776665
2: -4.9817009, 16.1492329, -5.2298994, 16.9050446, -21.8867455, 21.3791313
3: -7.0482631, 17.5268631, -7.4067106, 18.3266754, -25.3749352, 24.9335747
4: -8.3357964, 15.3324518, -8.7091722, 16.0773907, -24.4131851, 24.0416241

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1658737
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1658737
time: 0.48 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.0743117, 20.9155369, -4.1008406, 14.8422709, -20.9165821, 25.0163746
1: -6.9165897, 24.2520409, -4.6457920, 17.1946945, -24.1112843, 28.8978310
2: -7.5073729, 23.5865192, -5.1589270, 16.6772480, -24.1846199, 28.7454453
3: -10.6918325, 25.4853954, -7.2909489, 18.0929794, -28.7848129, 32.7763443
4: -12.1613083, 22.6388607, -8.6035089, 15.8505039, -28.0118122, 31.2423687

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1658737
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1658737
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.9600012, 14.3796635, -4.2527108, 15.3104954, -19.2704906, 18.6323738
1: -4.4894509, 16.6531467, -4.8148193, 17.7463245, -22.2357750, 21.4679661
2: -4.9817009, 16.1492329, -5.3409400, 17.2112198, -22.1929207, 21.4901733
3: -7.0482631, 17.5268631, -7.5317721, 18.6622658, -25.7105274, 25.0586357
4: -8.3357964, 15.3324518, -8.8748684, 16.3677406, -24.7035370, 24.2073174

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1669025, upper bound: 47.1702607
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1669025, upper bound: 47.1702607
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.0743117, 20.9155369, -4.1864662, 15.0959291, -21.1702404, 25.1020012
1: -6.9165897, 24.2520409, -4.7318096, 17.5026073, -24.4191933, 28.9838505
2: -7.5073729, 23.5865192, -5.2641234, 16.9664192, -24.4737911, 28.8506374
3: -10.6918325, 25.4853954, -7.4095511, 18.4113407, -29.1031723, 32.8949471
4: -12.1613083, 22.6388607, -8.7605743, 16.1251717, -28.2864799, 31.3994312

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1669025, upper bound: 47.1702607
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1669025, upper bound: 47.1702607
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.0656214, 14.6907520, -4.1611032, 15.0402527, -19.1058731, 18.8518562
1: -4.5916448, 17.0283775, -4.7245216, 17.4201145, -22.0117569, 21.7528992
2: -5.1096640, 16.4983921, -5.2298994, 16.9050446, -22.0147095, 21.7282906
3: -7.1945591, 17.9123573, -7.4067106, 18.3266754, -25.5212307, 25.3190689
4: -8.5250015, 15.6656189, -8.7091722, 16.0773907, -24.6023884, 24.3747864

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1658737
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.1992731, 21.2122402, -4.1008406, 14.8422709, -21.0415401, 25.3130798
1: -7.0349321, 24.6053638, -4.6457920, 17.1946945, -24.2296257, 29.2511559
2: -7.6394072, 23.9174843, -5.1589270, 16.6772480, -24.3166542, 29.0764122
3: -10.8425484, 25.8447590, -7.2909489, 18.0929794, -28.9355278, 33.1357079
4: -12.3402233, 22.9567204, -8.6035089, 15.8505039, -28.1907272, 31.5602303

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.0656214, 14.6907520, -4.2527108, 15.3104954, -19.3761120, 18.9434624
1: -4.5916448, 17.0283775, -4.8148193, 17.7463245, -22.3379669, 21.8431969
2: -5.1096640, 16.4983921, -5.3409400, 17.2112198, -22.3208847, 21.8393326
3: -7.1945591, 17.9123573, -7.5317721, 18.6622658, -25.8568249, 25.4441299
4: -8.5250015, 15.6656189, -8.8748684, 16.3677406, -24.8927422, 24.5404797

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1658737
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1712896
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.1992731, 21.2122402, -4.1864662, 15.0959291, -21.2951984, 25.3987064
1: -7.0349321, 24.6053638, -4.7318096, 17.5026073, -24.5375366, 29.3371735
2: -7.6394072, 23.9174843, -5.2641234, 16.9664192, -24.6058254, 29.1816025
3: -10.8425484, 25.8447590, -7.4095511, 18.4113407, -29.2538891, 33.2543068
4: -12.3402233, 22.9567204, -8.7605743, 16.1251717, -28.4653950, 31.7172928

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1683139, upper bound: 47.1655877
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1712896
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1712896
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.9600012, 14.3796635, -12.5386314, 43.3358650, -47.1625824, 26.9182930
1: -4.4894509, 16.6531467, -14.7782555, 50.4466515, -54.7677078, 31.4314022
2: -4.9817009, 16.1492329, -15.4454908, 49.2435837, -54.0856857, 31.5947208
3: -7.0482631, 17.5268631, -22.6835632, 53.1199379, -60.0255432, 40.2104225
4: -8.3357964, 15.3324518, -24.8940659, 47.4709549, -55.8067513, 40.2265053

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
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
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1663147, upper bound: 47.1635280
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1663147, upper bound: 47.1636863
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.0743117, 20.9155369, -12.5386314, 43.3358650, -49.2713814, 33.4541588
1: -6.9165897, 24.2520409, -14.7782555, 50.4466515, -57.1826744, 39.0302887
2: -7.5073729, 23.5865192, -15.4454908, 49.2435837, -56.6116486, 39.0320091
3: -10.6918325, 25.4853954, -22.6835632, 53.1199379, -63.6622887, 48.1689606
4: -12.1613083, 22.6388607, -24.8940659, 47.4709549, -59.6264420, 47.5329285

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1663147, upper bound: 47.1635280
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1663147, upper bound: 47.1636863
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.9463775, 14.3346872, -14.5672054, 49.6573448, -53.4374466, 28.9018898
1: -4.4730644, 16.6013870, -17.1258297, 57.8160286, -62.0779533, 33.7272186
2: -4.9651532, 16.0977135, -17.8585300, 56.4359894, -61.2211151, 33.9562416
3: -7.0233083, 17.4727135, -26.1972809, 60.8432045, -67.6776810, 43.6699944
4: -8.3106852, 15.2818718, -28.5813999, 54.4767380, -62.7684479, 43.8632736

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658992, upper bound: 47.1641507
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658992, upper bound: 47.1667574
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.0531664, 20.8245430, -14.5672054, 49.6573448, -55.5375977, 35.3917465
1: -6.8937969, 24.1513729, -17.1258297, 57.8160286, -64.4865494, 41.2771950
2: -7.4937367, 23.4771976, -17.8585300, 56.4359894, -63.7499084, 41.3357277
3: -10.6671381, 25.4023323, -26.1972809, 60.8432045, -71.3147049, 51.5996017
4: -12.1349783, 22.5499516, -28.5813999, 54.4767380, -66.5771637, 51.1313477

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658992, upper bound: 47.1639775
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658992, upper bound: 47.1644260
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.0656214, 14.6907520, -12.4907398, 43.1770287, -47.1120567, 27.1814919
1: -4.5916448, 17.0283775, -14.7196579, 50.2626801, -54.6885185, 31.7480316
2: -5.1096640, 16.4983921, -15.3879776, 49.0610275, -54.0328331, 31.8863697
3: -7.1945591, 17.9123573, -22.5956783, 52.9271393, -59.9821091, 40.5080338
4: -8.5250015, 15.6656189, -24.8065910, 47.2923355, -55.8173294, 40.4722099

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1706675, upper bound: 47.1645569
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1706675, upper bound: 47.1647151
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.1992731, 21.2122402, -12.4907398, 43.1770287, -49.2329178, 33.7029800
1: -7.0349321, 24.6053638, -14.7196579, 50.2626801, -57.1130753, 39.3250198
2: -7.6394072, 23.9174843, -15.3879776, 49.0610275, -56.5631447, 39.3054619
3: -10.8425484, 25.8447590, -22.5956783, 52.9271393, -63.6219254, 48.4404373
4: -12.3402233, 22.9567204, -24.8065910, 47.2923355, -59.6319542, 47.7633095

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1706675, upper bound: 47.1645569
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1706675, upper bound: 47.1647151
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -3.8997188, 14.2158442, -14.5215263, 49.5060539, -53.2372704, 28.7373695
1: -4.3987293, 16.4837189, -17.0701180, 57.6409378, -61.8283119, 33.5538292
2: -4.9164629, 15.9700289, -17.8036995, 56.2622681, -60.9972229, 33.7737274
3: -6.9077697, 17.3433609, -26.1135445, 60.6597366, -67.3806992, 43.4569054
4: -8.2416887, 15.1389160, -28.4977646, 54.3065796, -62.5296097, 43.6366806

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1610416, upper bound: 47.1623768
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1609134, upper bound: 47.1593715
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -3.9654596, 14.4745207, -14.5009260, 49.4402237, -53.2428856, 28.9754467
1: -4.4906955, 16.7857437, -17.0462151, 57.5644875, -61.8445740, 33.8319550
2: -4.9944859, 16.2698708, -17.7789803, 56.1873550, -61.0033760, 34.0488434
3: -7.0506811, 17.6513176, -26.0777988, 60.5800514, -67.4448547, 43.7291183
4: -8.3667002, 15.4228601, -28.4603119, 54.2336426, -62.5933838, 43.8831635

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1685466, upper bound: 47.1624788
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1685466, upper bound: 47.1652021
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.4524336, 43.1678200, -3.7945487, 13.8754206, -26.3278542, 46.8126564
1: -14.7858372, 50.1457520, -4.2991891, 16.0663052, -30.8521423, 54.2637062
2: -15.2835922, 49.0636292, -4.7751355, 15.5799980, -30.8635902, 53.6851463
3: -22.5760632, 52.7194977, -6.7621970, 16.8902683, -39.4663200, 59.3126564
4: -24.4824944, 47.3548775, -8.0042782, 14.7790022, -39.2614975, 55.3345718

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1537655, upper bound: 47.1626016
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1530379, upper bound: 47.1684371
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.5043392, 43.3323669, -3.7945487, 13.8754206, -26.3797607, 46.9887505
1: -14.8310118, 50.3433380, -4.2991891, 16.0663052, -30.8973160, 54.4775581
2: -15.3490219, 49.2404137, -4.7751355, 15.5799980, -30.9290199, 53.8756943
3: -22.6377640, 52.9216881, -6.7621970, 16.8902683, -39.5280304, 59.5320549
4: -24.5850430, 47.5082436, -8.0042782, 14.7790022, -39.3640442, 55.4980736

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1537655, upper bound: 47.1644166
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1530379, upper bound: 47.1701913
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -12.4524336, 43.1678200, -3.8281255, 13.9594917, -26.4119244, 46.8489799
1: -14.7858372, 50.1457520, -4.3222551, 16.1841278, -30.9699650, 54.2875023
2: -15.2835922, 49.0636292, -4.8204536, 15.6792259, -30.9628181, 53.7312393
3: -22.5760632, 52.7194977, -6.7869787, 17.0132713, -39.5893326, 59.3397560
4: -24.4824944, 47.3548775, -8.0733624, 14.8673763, -39.3498611, 55.4083138

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1561138, upper bound: 47.1706184
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1542796, upper bound: 47.1660953
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.5043392, 43.3323669, -3.8281255, 13.9594917, -26.4638271, 47.0250778
1: -14.8310118, 50.3433380, -4.3222551, 16.1841278, -31.0151405, 54.5013542
2: -15.3490219, 49.2404137, -4.8204536, 15.6792259, -31.0282478, 53.9217873
3: -22.6377640, 52.9216881, -6.7869787, 17.0132713, -39.6510353, 59.5591545
4: -24.5850430, 47.5082436, -8.0733624, 14.8673763, -39.4524193, 55.5718155

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1561138, upper bound: 47.1731003
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1542796, upper bound: 47.1676738
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.3346949, 46.0455627, -4.1578374, 15.0287895, -28.3634834, 50.0830002
1: -15.7996721, 53.5419350, -4.7208223, 17.4065552, -33.2062263, 58.1066589
2: -16.3717403, 52.3268585, -5.2258611, 16.8920612, -33.2637978, 57.4237022
3: -24.1192474, 56.3102837, -7.4009533, 18.3124847, -42.4317322, 63.5778542
4: -26.2549019, 50.4941635, -8.7025156, 16.0651093, -42.3200111, 59.1966782

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1630553
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1630553
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -15.4945087, 52.7464104, -4.0976014, 14.8309727, -30.3254795, 56.6916504
1: -18.2942677, 61.3451691, -4.6421185, 17.1813221, -35.4755859, 65.7883148
2: -18.9236336, 59.9523926, -5.1549048, 16.6644688, -35.5881004, 64.9397430
3: -27.8473854, 64.4847412, -7.2852187, 18.0789375, -45.9263153, 71.5932693
4: -30.1493359, 57.9222450, -8.5968904, 15.8383808, -45.9877167, 66.5055389

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630553
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630553
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.2768965, 45.8559189, -4.2501745, 15.3010006, -28.5778961, 49.9885063
1: -15.7296848, 53.3226051, -4.8118176, 17.7350483, -33.4647331, 57.9803848
2: -16.3027439, 52.1096497, -5.3376241, 17.2004509, -33.5031967, 57.3197517
3: -24.0152550, 56.0807190, -7.5270677, 18.6503677, -42.6656189, 63.4769974
4: -26.1505356, 50.2819443, -8.8692446, 16.3576317, -42.5081673, 59.1511879

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603529, upper bound: 47.1677334
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603529, upper bound: 47.1677334
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -15.4377794, 52.5606575, -4.1838465, 15.0865030, -30.5242786, 56.5951462
1: -18.2258282, 61.1304893, -4.7288284, 17.4914265, -35.7172508, 65.6622467
2: -18.8561401, 59.7396584, -5.2608356, 16.9557152, -35.8118553, 64.8345871
3: -27.7454967, 64.2600403, -7.4048800, 18.3995342, -46.1450233, 71.4910736
4: -30.0468006, 57.7142754, -8.7549877, 16.1151276, -46.1619225, 66.4617157

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1593715, upper bound: 47.1609134
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1615719, upper bound: 47.1677334
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1615719, upper bound: 47.1677334
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.5986252, 43.6937981, -12.0762205, 41.8052101, -54.2470856, 55.6425209
1: -14.9788866, 50.7203484, -14.2236443, 48.6441841, -63.3786087, 64.7199402
2: -15.4106741, 49.6562500, -14.8743973, 47.4772987, -62.6779060, 64.3309784
3: -22.8058186, 53.2327690, -21.8283043, 51.2009354, -73.7309875, 74.7939148
4: -24.5636044, 47.8904190, -23.9546871, 45.7445297, -70.1455765, 71.6803970

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0629257, upper bound: 47.1213464
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0629019, upper bound: 47.1212286
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.3208189, 42.8074799, -12.2238817, 42.2903328, -54.4650002, 54.9098015
1: -14.6402922, 49.7488747, -14.4026480, 49.2146339, -63.6133614, 63.9334297
2: -15.1311321, 48.6670380, -15.0601454, 48.0383606, -62.9628792, 63.5343323
3: -22.3571396, 52.3079605, -22.1064796, 51.8227615, -73.9097824, 74.1553497
4: -24.2758942, 46.9409256, -24.2737064, 46.3016548, -70.4192886, 71.0636063

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0923033, upper bound: 47.0748229
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0923033, upper bound: 47.1575457
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -12.5221500, 43.4309578, -14.2004910, 48.4073105, -60.7270851, 57.4871407
1: -14.8811102, 50.4199944, -16.6680126, 56.3274155, -70.8968887, 66.8606873
2: -15.3186026, 49.3529739, -17.3876972, 54.9783020, -70.0253906, 66.5328522
3: -22.6617718, 52.9194412, -25.4889965, 59.2506905, -81.5713348, 78.1284561
4: -24.4265766, 47.5942345, -27.8005390, 53.0436440, -77.2665558, 75.2171555

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0629325, upper bound: 47.1258307
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0627673, upper bound: 47.1211887
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.2366409, 42.5208817, -14.3508892, 48.9078751, -60.9527588, 56.7371025
1: -14.5335102, 49.4203873, -16.8567600, 56.9158020, -71.1408463, 66.0566483
2: -15.0311060, 48.3372574, -17.5816803, 55.5580559, -70.3214188, 65.7189178
3: -22.1995735, 51.9658394, -25.7801628, 59.8938713, -81.7580032, 77.4747086
4: -24.1262417, 46.6197205, -28.1299934, 53.6215401, -77.5485916, 74.5862503

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1282700, upper bound: 47.1569570
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1275519, upper bound: 47.1398334
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.3346949, 46.0455627, -12.7497654, 43.9991837, -57.2137756, 58.6866570
1: -15.7996721, 53.5419350, -15.0207338, 51.2055168, -66.7977905, 68.3629150
2: -16.3717403, 52.3268585, -15.6935768, 49.9844093, -66.1728058, 67.8436127
3: -24.1192474, 56.3102837, -23.0376320, 53.9141006, -77.8037949, 79.1169510
4: -26.2549019, 50.4941635, -25.2652245, 48.1889725, -74.3189926, 75.6311646

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1002878, upper bound: 47.0815967
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1577159, upper bound: 47.1575741
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -15.4945087, 52.7464104, -12.7497654, 43.9991837, -59.3634415, 65.3558578
1: -18.2942677, 61.3451691, -15.0207338, 51.2055168, -69.2900238, 76.1238174
2: -18.9236336, 59.9523926, -15.6935768, 49.9844093, -68.7208099, 75.4303589
3: -27.8473854, 64.4847412, -23.0376320, 53.9141006, -81.5218506, 87.2479706
4: -30.1493359, 57.9222450, -25.2652245, 48.1889725, -78.2027893, 83.0333023

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1002878, upper bound: 47.0815967
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1577159, upper bound: 47.1575741
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.3346949, 46.0455627, -14.8889523, 50.6483345, -63.8300095, 60.8149872
1: -15.7996721, 53.5419350, -17.4876347, 58.9481316, -74.4971161, 70.8289108
2: -16.3717403, 52.3268585, -18.2277164, 57.5445175, -73.6912460, 70.3727493
3: -24.1192474, 56.3102837, -26.7319660, 62.0291901, -85.8720932, 82.8000946
4: -26.2549019, 50.4941635, -29.1401997, 55.5493774, -81.6510925, 79.4941559

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1459795, upper bound: 47.1613670
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1441747, upper bound: 47.1453345
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -15.4945087, 52.7464104, -14.8889523, 50.6483345, -65.9796829, 67.4842148
1: -18.2942677, 61.3451691, -17.4876347, 58.9481316, -76.9893494, 78.5898056
2: -18.9236336, 59.9523926, -18.2277164, 57.5445175, -76.2392578, 77.9595032
3: -27.8473854, 64.4847412, -26.7319660, 62.0291901, -89.5901566, 90.9311218
4: -30.1493359, 57.9222450, -29.1401997, 55.5493774, -85.5348816, 86.8962860

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1459795, upper bound: 47.1607871
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1441747, upper bound: 47.1453345
time: 0.72 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.92 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1658737
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1658737
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1658737
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1658737
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1669025, upper bound: 47.1702607
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1669025, upper bound: 47.1702607
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1669025, upper bound: 47.1702607
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1669025, upper bound: 47.1702607
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1658737
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1658737
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1712896
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1712896
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1712896
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1663147, upper bound: 47.1635280
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1663147, upper bound: 47.1636863
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1663147, upper bound: 47.1635280
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1663147, upper bound: 47.1636863
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1658992, upper bound: 47.1641507
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1658992, upper bound: 47.1667574
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1658992, upper bound: 47.1639775
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1658992, upper bound: 47.1644260
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1706675, upper bound: 47.1645569
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1706675, upper bound: 47.1647151
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1706675, upper bound: 47.1645569
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1706675, upper bound: 47.1647151
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1610416, upper bound: 47.1623768
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1609134, upper bound: 47.1593715
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1685466, upper bound: 47.1624788
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1685466, upper bound: 47.1652021
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1537655, upper bound: 47.1626016
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1530379, upper bound: 47.1684371
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1537655, upper bound: 47.1644166
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1530379, upper bound: 47.1701913
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1561138, upper bound: 47.1706184
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1542796, upper bound: 47.1660953
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1561138, upper bound: 47.1731003
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1542796, upper bound: 47.1676738
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1630553
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1630553
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630553
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630553
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1603529, upper bound: 47.1677334
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1603529, upper bound: 47.1677334
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1615719, upper bound: 47.1677334
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1615719, upper bound: 47.1677334
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.0629257, upper bound: 47.1213464
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.0629019, upper bound: 47.1212286
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.0923033, upper bound: 47.0748229
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.0923033, upper bound: 47.1575457
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.0629325, upper bound: 47.1258307
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.0627673, upper bound: 47.1211887
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1282700, upper bound: 47.1569570
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1275519, upper bound: 47.1398334
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1002878, upper bound: 47.0815967
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1577159, upper bound: 47.1575741
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1002878, upper bound: 47.0815967
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1577159, upper bound: 47.1575741
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1459795, upper bound: 47.1613670
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1441747, upper bound: 47.1453345
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1459795, upper bound: 47.1607871
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 4, lower bound: -47.1441747, upper bound: 47.1453345

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.9600012, 14.3796635, -3.9600012, 14.3796635, -18.3396645, 18.3396645
1: -4.4894509, 16.6531467, -4.4894509, 16.6531467, -21.1425972, 21.1425972
2: -4.9817009, 16.1492329, -4.9817009, 16.1492329, -21.1309319, 21.1309319
3: -7.0482631, 17.5268631, -7.0482631, 17.5268631, -24.5751228, 24.5751228
4: -8.3357964, 15.3324518, -8.3357964, 15.3324518, -23.6682472, 23.6682472

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1667343, upper bound: 47.1655052
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1720608, upper bound: 47.1649248
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.9600012, 14.3796635, -6.0743117, 20.9155369, -24.8755360, 20.4539757
1: -4.4894509, 16.6531467, -6.9165897, 24.2520409, -28.7414894, 23.5697327
2: -4.9817009, 16.1492329, -7.5073729, 23.5865192, -28.5682201, 23.6566029
3: -7.0482631, 17.5268631, -10.6918325, 25.4853954, -32.5336533, 28.2186947
4: -8.3357964, 15.3324518, -12.1613083, 22.6388607, -30.9746571, 27.4937592

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1667343, upper bound: 47.1655052
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1720608, upper bound: 47.1649248
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.0743117, 20.9155369, -3.9463775, 14.3346872, -20.4089985, 24.8619137
1: -6.9165897, 24.2520409, -4.4730644, 16.6013870, -23.5179729, 28.7251015
2: -7.5073729, 23.5865192, -4.9651532, 16.0977135, -23.6050854, 28.5516720
3: -10.6918325, 25.4853954, -7.0233083, 17.4727135, -28.1645451, 32.5087051
4: -12.1613083, 22.6388607, -8.3106852, 15.2818718, -27.4431801, 30.9495430

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 31

Time for candidate selection: 4.47 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1653200, upper bound: 47.1649650
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1651267, upper bound: 47.1651267
time: 0.49 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.0743117, 20.9155369, -6.0531664, 20.8245430, -26.8988533, 26.9687042
1: -6.9165897, 24.2520409, -6.8937969, 24.1513729, -31.0679569, 31.1458321
2: -7.5073729, 23.5865192, -7.4937367, 23.4771976, -30.9845695, 31.0802536
3: -10.6918325, 25.4853954, -10.6671381, 25.4023323, -36.0941544, 36.1525345
4: -12.1613083, 22.6388607, -12.1349783, 22.5499516, -34.7112579, 34.7738380

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 31

Time for candidate selection: 4.39 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1653200, upper bound: 47.1649650
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1651267, upper bound: 47.1651267
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.9600012, 14.3796635, -4.0656214, 14.6907520, -18.6507530, 18.4452858
1: -4.4894509, 16.6531467, -4.5916448, 17.0283775, -21.5178280, 21.2447891
2: -4.9817009, 16.1492329, -5.1096640, 16.4983921, -21.4800930, 21.2588959
3: -7.0482631, 17.5268631, -7.1945591, 17.9123573, -24.9606171, 24.7214203
4: -8.3357964, 15.3324518, -8.5250015, 15.6656189, -24.0014153, 23.8574524

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1680141, upper bound: 47.1700342
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1680141, upper bound: 47.1697033
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.9600012, 14.3796635, -6.1992731, 21.2122402, -25.1722393, 20.5789337
1: -4.4894509, 16.6531467, -7.0349321, 24.6053638, -29.0948143, 23.6880741
2: -4.9817009, 16.1492329, -7.6394072, 23.9174843, -28.8991852, 23.7886391
3: -7.0482631, 17.5268631, -10.8425484, 25.8447590, -32.8930130, 28.3694096
4: -8.3357964, 15.3324518, -12.3402233, 22.9567204, -31.2925148, 27.6726761

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1667343, upper bound: 47.1700342
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1733507, upper bound: 47.1697033
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.0743117, 20.9155369, -4.0656214, 14.6907520, -20.7650642, 24.9811573
1: -6.9165897, 24.2520409, -4.5916448, 17.0283775, -23.9449654, 28.8436813
2: -7.5073729, 23.5865192, -5.1096640, 16.4983921, -24.0057640, 28.6961823
3: -10.6918325, 25.4853954, -7.1945591, 17.9123573, -28.6041908, 32.6799545
4: -12.1613083, 22.6388607, -8.5250015, 15.6656189, -27.8269253, 31.1638622

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 31

Time for candidate selection: 4.39 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1662376, upper bound: 47.1695454
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660443, upper bound: 47.1697070
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.0743117, 20.9155369, -6.1992731, 21.2122402, -27.2865505, 27.1148052
1: -6.9165897, 24.2520409, -7.0349321, 24.6053638, -31.5219517, 31.2869682
2: -7.5073729, 23.5865192, -7.6394072, 23.9174843, -31.4248581, 31.2259254
3: -10.6918325, 25.4853954, -10.8425484, 25.8447590, -36.5365906, 36.3279419
4: -12.1613083, 22.6388607, -12.3402233, 22.9567204, -35.1180267, 34.9790840

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 31

Time for candidate selection: 4.39 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1662376, upper bound: 47.1695454
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660443, upper bound: 47.1697070
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.0656214, 14.6907520, -3.9600012, 14.3796635, -18.4452858, 18.6507530
1: -4.5916448, 17.0283775, -4.4894509, 16.6531467, -21.2447891, 21.5178280
2: -5.1096640, 16.4983921, -4.9817009, 16.1492329, -21.2588959, 21.4800930
3: -7.1945591, 17.9123573, -7.0482631, 17.5268631, -24.7214203, 24.9606171
4: -8.5250015, 15.6656189, -8.3357964, 15.3324518, -23.8574524, 24.0014133

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1693558, upper bound: 47.1657537
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1722610, upper bound: 47.1649256
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.0656214, 14.6907520, -6.0743117, 20.9155369, -24.9811573, 20.7650642
1: -4.5916448, 17.0283775, -6.9165897, 24.2520409, -28.8436832, 23.9449654
2: -5.1096640, 16.4983921, -7.5073729, 23.5865192, -28.6961823, 24.0057640
3: -7.1945591, 17.9123573, -10.6918325, 25.4853954, -32.6799545, 28.6041908
4: -8.5250015, 15.6656189, -12.1613083, 22.6388607, -31.1638622, 27.8269253

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1693558, upper bound: 47.1657537
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1691523, upper bound: 47.1649256
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.1992731, 21.2122402, -3.9463775, 14.3346872, -20.5339584, 25.1586170
1: -7.0349321, 24.6053638, -4.4730644, 16.6013870, -23.6363182, 29.0784283
2: -7.6394072, 23.9174843, -4.9651532, 16.0977135, -23.7371197, 28.8826370
3: -10.8425484, 25.8447590, -7.0233083, 17.4727135, -28.3152618, 32.8680687
4: -12.3402233, 22.9567204, -8.3106852, 15.2818718, -27.6220951, 31.2674007

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1652912, upper bound: 47.1649233
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1686617, upper bound: 47.1643232
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.1992731, 21.2122402, -6.0531664, 20.8245430, -27.0238132, 27.2654076
1: -7.0349321, 24.6053638, -6.8937969, 24.1513729, -31.1863041, 31.4991512
2: -7.6394072, 23.9174843, -7.4937367, 23.4771976, -31.1166039, 31.4112206
3: -10.8425484, 25.8447590, -10.6671381, 25.4023323, -36.2448692, 36.5118980
4: -12.3402233, 22.9567204, -12.1349783, 22.5499516, -34.8901749, 35.0916977

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 31

Time for candidate selection: 4.66 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1697070, upper bound: 47.1653558
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1697070, upper bound: 47.1660443
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.0656214, 14.6907520, -4.0656214, 14.6907520, -18.7563744, 18.7563744
1: -4.5916448, 17.0283775, -4.5916448, 17.0283775, -21.6200199, 21.6200199
2: -5.1096640, 16.4983921, -5.1096640, 16.4983921, -21.6080551, 21.6080551
3: -7.1945591, 17.9123573, -7.1945591, 17.9123573, -25.1069145, 25.1069145
4: -8.5250015, 15.6656189, -8.5250015, 15.6656189, -24.1906185, 24.1906166

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1667343, upper bound: 47.1704447
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1726404, upper bound: 47.1697029
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.0656214, 14.6907520, -6.1992731, 21.2122402, -25.2778625, 20.8900261
1: -4.5916448, 17.0283775, -7.0349321, 24.6053638, -29.1970081, 24.0633087
2: -5.1096640, 16.4983921, -7.6394072, 23.9174843, -29.0271492, 24.1377983
3: -7.1945591, 17.9123573, -10.8425484, 25.8447590, -33.0393143, 28.7549057
4: -8.5250015, 15.6656189, -12.3402233, 22.9567204, -31.4817200, 28.0058403

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1705002, upper bound: 47.1704447
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1726404, upper bound: 47.1697029
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.1992731, 21.2122402, -4.0656214, 14.6907520, -20.8900261, 25.2778625
1: -7.0349321, 24.6053638, -4.5916448, 17.0283775, -24.0633087, 29.1970062
2: -7.6394072, 23.9174843, -5.1096640, 16.4983921, -24.1377983, 29.0271492
3: -10.8425484, 25.8447590, -7.1945591, 17.9123573, -28.7549057, 33.0393143
4: -12.3402233, 22.9567204, -8.5250015, 15.6656189, -28.0058403, 31.4817200

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1655877, upper bound: 47.1695583
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1688906, upper bound: 47.1689582
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.1992731, 21.2122402, -6.1992731, 21.2122402, -27.4115124, 27.4115124
1: -7.0349321, 24.6053638, -7.0349321, 24.6053638, -31.6402969, 31.6402912
2: -7.6394072, 23.9174843, -7.6394072, 23.9174843, -31.5568924, 31.5568924
3: -10.8425484, 25.8447590, -10.8425484, 25.8447590, -36.6873055, 36.6873055
4: -12.3402233, 22.9567204, -12.3402233, 22.9567204, -35.2969437, 35.2969398

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1655877, upper bound: 47.1695583
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1655877, upper bound: 47.1689582
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.9600012, 14.3796635, -12.3423586, 42.6287956, -46.4451675, 26.7220230
1: -4.4894509, 16.6531467, -14.5416908, 49.6136169, -53.9217567, 31.1948376
2: -4.9817009, 16.1492329, -15.2078686, 48.4294510, -53.2578964, 31.3570976
3: -7.0482631, 17.5268631, -22.3209610, 52.2472763, -59.1398773, 39.8478241
4: -8.3357964, 15.3324518, -24.5104485, 46.6945839, -55.0303802, 39.8428955

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0840400, upper bound: 47.0845888
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0840400, upper bound: 47.1616874
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.9600012, 14.3796635, -12.4299145, 42.9139175, -46.7424431, 26.8095741
1: -4.4894509, 16.6531467, -14.6292362, 49.9546661, -54.2797432, 31.2823830
2: -4.9817009, 16.1492329, -15.3131676, 48.7416992, -53.5831909, 31.4624004
3: -7.0482631, 17.5268631, -22.4420071, 52.5942459, -59.5046768, 39.9688644
4: -8.3357964, 15.3324518, -24.6697426, 46.9733429, -55.3091393, 40.0021935

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0840400, upper bound: 47.0845888
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1711989, upper bound: 47.1618179
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.0743117, 20.9155369, -12.3423586, 42.6287956, -48.5539665, 33.2578888
1: -6.9165897, 24.2520409, -14.5416908, 49.6136169, -56.3367233, 38.7937241
2: -7.5073729, 23.5865192, -15.2078686, 48.4294510, -55.7838593, 38.7943802
3: -10.6918325, 25.4853954, -22.3209610, 52.2472763, -62.7766266, 47.8063583
4: -12.1613083, 22.6388607, -24.5104485, 46.6945839, -58.8413239, 47.1493073

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1663147, upper bound: 47.1594681
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 31

Time for candidate selection: 5.42 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1659227, upper bound: 47.1628127
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1657293, upper bound: 47.1629743
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.0743117, 20.9155369, -12.4299145, 42.9139175, -48.8512421, 33.3454399
1: -6.9165897, 24.2520409, -14.6292362, 49.9546661, -56.6947060, 38.8812637
2: -7.5073729, 23.5865192, -15.3131676, 48.7416992, -56.1091537, 38.8996849
3: -10.6918325, 25.4853954, -22.4420071, 52.5942459, -63.1414261, 47.9274025
4: -12.1613083, 22.6388607, -24.6697426, 46.9733429, -59.1321068, 47.3086014

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1663147, upper bound: 47.1594681
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 31

Time for candidate selection: 5.13 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1659227, upper bound: 47.1629710
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1657293, upper bound: 47.1631326
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.9463775, 14.3346872, -14.3869314, 49.0037346, -52.7754936, 28.7216187
1: -4.4730644, 16.6013870, -16.9090824, 57.0448570, -61.2964706, 33.5104675
2: -4.9651532, 16.0977135, -17.6411133, 55.6844215, -60.4604301, 33.7388191
3: -7.0233083, 17.4727135, -25.8681087, 60.0354729, -66.8605652, 43.3408203
4: -8.3106852, 15.2818718, -28.2291660, 53.7631760, -62.0490761, 43.5110397

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660089, upper bound: 47.1635700
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712630, upper bound: 47.1629780
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.9463775, 14.3346872, -14.4889946, 49.2915764, -53.0760880, 28.8236809
1: -4.4730644, 16.6013870, -17.0001602, 57.3875656, -61.6560402, 33.6015472
2: -4.9651532, 16.0977135, -17.7474689, 55.9970512, -60.7871857, 33.8451767
3: -7.0233083, 17.4727135, -25.9868507, 60.3842926, -67.2277451, 43.4595642
4: -8.3106852, 15.2818718, -28.3873749, 54.0358543, -62.3328209, 43.6692467

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660089, upper bound: 47.1659541
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712630, upper bound: 47.1656585
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.0531664, 20.8245430, -14.3869314, 49.0037346, -54.8756447, 35.2114716
1: -6.8937969, 24.1513729, -16.9090824, 57.0448570, -63.7050667, 41.0604553
2: -7.4937367, 23.4771976, -17.6411133, 55.6844215, -62.9892120, 41.1183090
3: -10.6671381, 25.4023323, -25.8681087, 60.0354729, -70.4975739, 51.2704353
4: -12.1349783, 22.5499516, -28.2291660, 53.7631760, -65.8577957, 50.7791138

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1655820, upper bound: 47.1490277
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 31

Time for candidate selection: 5.24 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1655471, upper bound: 47.1632665
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1653537, upper bound: 47.1633916
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.0531664, 20.8245430, -14.4889946, 49.2915764, -55.1762390, 35.3135376
1: -6.8937969, 24.1513729, -17.0001602, 57.3875656, -64.0646439, 41.1515236
2: -7.4937367, 23.4771976, -17.7474689, 55.9970512, -63.3159676, 41.2246666
3: -10.6671381, 25.4023323, -25.9868507, 60.3842926, -70.8647766, 51.3891754
4: -12.1349783, 22.5499516, -28.3873749, 54.0358543, -66.1415634, 50.9373245

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1655820, upper bound: 47.1490449
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 31

Time for candidate selection: 5.35 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1655471, upper bound: 47.1637518
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1653537, upper bound: 47.1638769
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.0656214, 14.6907520, -12.3423586, 42.6287956, -46.5529938, 27.0331116
1: -4.5916448, 17.0283775, -14.5416908, 49.6136169, -54.0258064, 31.5700665
2: -5.1096640, 16.4983921, -15.2078686, 48.4294510, -53.3872452, 31.7062588
3: -7.1945591, 17.9123573, -22.3209610, 52.2472763, -59.2887459, 40.2333183
4: -8.5250015, 15.6656189, -24.5104485, 46.6945839, -55.2195778, 40.1760674

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1689331, upper bound: 47.1627049
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1718718, upper bound: 47.1618518
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.0656214, 14.6907520, -12.4299145, 42.9139175, -46.8502693, 27.1206627
1: -4.5916448, 17.0283775, -14.6292362, 49.9546661, -54.3837967, 31.6576099
2: -5.1096640, 16.4983921, -15.3131676, 48.7416992, -53.7125435, 31.8115597
3: -7.1945591, 17.9123573, -22.4420071, 52.5942459, -59.6535416, 40.3543587
4: -8.5250015, 15.6656189, -24.6697426, 46.9733429, -55.4983406, 40.3353615

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1689331, upper bound: 47.1627049
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1718718, upper bound: 47.1618518
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.1992731, 21.2122402, -12.3423586, 42.6287956, -48.6738586, 33.5545998
1: -7.0349321, 24.6053638, -14.5416908, 49.6136169, -56.4503593, 39.1470566
2: -7.6394072, 23.9174843, -15.2078686, 48.4294510, -55.9175644, 39.1253471
3: -10.8425484, 25.8447590, -22.3209610, 52.2472763, -62.9285622, 48.1657181
4: -12.3402233, 22.9567204, -24.5104485, 46.6945839, -59.0249786, 47.4671707

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1666831, upper bound: 47.1603753
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1671889, upper bound: 47.1603529
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.1992731, 21.2122402, -12.4299145, 42.9139175, -48.9711304, 33.6421547
1: -7.0349321, 24.6053638, -14.6292362, 49.9546661, -56.8083458, 39.2345886
2: -7.6394072, 23.9174843, -15.3131676, 48.7416992, -56.2428551, 39.2306519
3: -10.8425484, 25.8447590, -22.4420071, 52.5942459, -63.2933578, 48.2867661
4: -12.3402233, 22.9567204, -24.6697426, 46.9733429, -59.3135643, 47.6264648

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1666831, upper bound: 47.1603753
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1671889, upper bound: 47.1603529
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.5023835, 12.9532480, -13.3971224, 45.9019775, -49.1933937, 26.3503704
1: -3.9405715, 15.0175991, -15.7876940, 53.3944893, -57.0747528, 30.8052940
2: -4.4274468, 14.5366096, -16.4233551, 52.1678848, -56.3685570, 30.9599628
3: -6.2139096, 15.7934027, -24.1539440, 56.1773872, -62.1546593, 39.9473457
4: -7.4878693, 13.7354240, -26.2845478, 50.3497658, -57.7670937, 40.0199585

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1608106, upper bound: 47.1593715
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1608106, upper bound: 47.1593715
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.8971307, 14.2062473, -14.3484478, 48.8987923, -52.6414528, 28.5546913
1: -4.3956432, 16.4724064, -16.8643131, 56.9298782, -61.1315613, 33.3367195
2: -4.9130492, 15.9591961, -17.5888729, 55.5692902, -60.3156013, 33.5480652
3: -6.9029498, 17.3313789, -25.7969761, 59.9153061, -66.6474228, 43.1283569
4: -8.2359772, 15.1286573, -28.1547699, 53.6416779, -61.8684998, 43.2834282

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1608933, upper bound: 47.1593715
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1608933, upper bound: 47.1586970
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -3.9654596, 14.4745207, -14.3660011, 48.9375038, -52.7313499, 28.8405228
1: -4.4906955, 16.7857437, -16.8850479, 56.9679031, -61.2370453, 33.6707878
2: -4.9944859, 16.2698708, -17.6162319, 55.6090546, -60.4154167, 33.8861008
3: -7.0506811, 17.6513176, -25.8321514, 59.9552841, -66.8099670, 43.4834671
4: -8.3667002, 15.4228601, -28.1915169, 53.6897774, -62.0432549, 43.6143723

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1685466, upper bound: 47.1624788
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1685466, upper bound: 47.1624055
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -3.9654596, 14.4745207, -14.4655561, 49.2181511, -53.0248337, 28.9400749
1: -4.4906955, 16.7857437, -16.9732742, 57.3023643, -61.5884705, 33.7590179
2: -4.9944859, 16.2698708, -17.7198067, 55.9134102, -60.7339783, 33.9896774
3: -7.0506811, 17.6513176, -25.9466171, 60.2954140, -67.1685410, 43.5979347
4: -8.3667002, 15.4228601, -28.3454285, 53.9540901, -62.3187675, 43.7682877

Time for backsubstitution: 1.85 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0833333, mid=0.0833333, abs_max=50.8192024230957
rel_dist={4: [-47.18088696914194, 47.18088696914194]}

## Binary search (step 2) starts
Candidate diff: 0.0416667


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

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
time: 0.91 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.57 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 4, lower bound: -47.1725005, upper bound: 47.1680812
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 4, lower bound: -47.1678244, upper bound: 47.1678244

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.3289332, 15.6493454, -5.8106618, 20.5728836, -24.9018173, 21.4600010
1: -4.9227071, 18.1375618, -6.6639028, 23.8333817, -28.7560883, 24.8014641
2: -5.4355350, 17.6055393, -7.2380991, 23.2434902, -28.6790257, 24.8436375
3: -7.7107296, 19.0719624, -10.3406324, 25.0188999, -32.7296295, 29.4125938
4: -9.0420084, 16.7446327, -11.7947159, 22.2706470, -31.3126507, 28.5393486

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21

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

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678244, upper bound: 47.1678244
time: 0.89 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678244, upper bound: 47.1678244
time: 0.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.22 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 4, lower bound: -47.1678244, upper bound: 47.1678244
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 4, lower bound: -47.1678244, upper bound: 47.1678244
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 4, lower bound: -47.1678244, upper bound: 47.1678244
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 4, lower bound: -47.1678244, upper bound: 47.1678244

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -4.3289332, 15.6493454, -4.3289332, 15.6493454, -19.9782753, 19.9782753
1: -4.9227071, 18.1375618, -4.9227071, 18.1375618, -23.0602684, 23.0602684
2: -5.4355350, 17.6055393, -5.4355350, 17.6055393, -23.0410748, 23.0410748
3: -7.7107296, 19.0719624, -7.7107296, 19.0719624, -26.7826920, 26.7826920
4: -9.0420084, 16.7446327, -9.0420084, 16.7446327, -25.7866364, 25.7866364

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712060, upper bound: 47.1672004
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712060, upper bound: 47.1680541
time: 0.89 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -4.3289332, 15.6493454, -12.4547701, 42.7791710, -46.9673615, 28.1041126
1: -4.9227071, 18.1375618, -14.6426277, 49.7638359, -54.5080185, 32.7801895
2: -5.4355350, 17.6055393, -15.3634157, 48.6813507, -53.9675903, 32.9689560
3: -7.7107296, 19.0719624, -22.4968510, 52.4056053, -59.9649010, 41.5688057
4: -9.0420084, 16.7446327, -24.6993065, 47.0198441, -56.0618477, 41.4439354

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712060, upper bound: 47.1672004
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712060, upper bound: 47.1680541
time: 0.52 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -13.8669882, 47.8264809, -4.3289332, 15.6493454, -29.5163288, 52.0300331
1: -16.4424095, 55.6123199, -4.9227071, 18.1375618, -34.5799713, 60.3761864
2: -17.0157623, 54.3669052, -5.4355350, 17.6055393, -34.6212921, 59.6714478
3: -25.0840302, 58.4735527, -7.7107296, 19.0719624, -44.1559906, 66.0495911
4: -27.2452755, 52.4822922, -9.0420084, 16.7446327, -43.9899063, 61.5242958

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1644321, upper bound: 47.1619489
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1621160, upper bound: 47.1621160
time: 0.54 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -13.8669882, 47.8264809, -13.0686092, 45.0560722, -58.8123169, 60.7855453
1: -16.4424095, 55.6123199, -15.4047651, 52.4332733, -68.6767883, 70.8160324
2: -17.0157623, 54.3669052, -16.0822449, 51.1983566, -68.0413132, 70.2737198
3: -25.0840302, 58.4735527, -23.6169319, 55.1978226, -80.0572128, 81.8583908
4: -27.2452755, 52.4822922, -25.8539028, 49.3803787, -76.5030136, 78.2091293

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1644321, upper bound: 47.1619489
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1621160, upper bound: 47.1621160
time: 0.81 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.30 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 4, lower bound: -47.1712060, upper bound: 47.1672004
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 4, lower bound: -47.1712060, upper bound: 47.1680541
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 4, lower bound: -47.1712060, upper bound: 47.1672004
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 4, lower bound: -47.1712060, upper bound: 47.1680541
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 4, lower bound: -47.1644321, upper bound: 47.1619489
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 4, lower bound: -47.1621160, upper bound: 47.1621160
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 4, lower bound: -47.1644321, upper bound: 47.1619489
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 4, lower bound: -47.1621160, upper bound: 47.1621160

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.1957455, 15.1547375, -4.3182654, 15.6102047, -19.8059502, 19.4729958
1: -4.7651372, 17.5528717, -4.9101896, 18.0906925, -22.8558292, 22.4630623
2: -5.2724891, 17.0362396, -5.4225030, 17.5597878, -22.8322735, 22.4587421
3: -7.4683981, 18.4647770, -7.6914511, 19.0231705, -26.4915676, 26.1562271
4: -8.7734575, 16.2064075, -9.0204601, 16.7017097, -25.4751663, 25.2268658

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1797884, upper bound: 47.1797884
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1797884, upper bound: 47.1799785
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.2849174, 15.4173374, -4.2537827, 15.4004431, -19.6853561, 19.6711197
1: -4.8534427, 17.8700066, -4.8296795, 17.8530769, -22.7065182, 22.6996861
2: -5.3807850, 17.3343678, -5.3447952, 17.3214970, -22.7022820, 22.6791630
3: -7.5899143, 18.7914963, -7.5715981, 18.7736073, -26.3635216, 26.3630943
4: -8.9349260, 16.4891949, -8.9037714, 16.4657497, -25.4006691, 25.3929672

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1673438, upper bound: 47.1715203
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712896, upper bound: 47.1712896
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.1957455, 15.1547375, -12.4400482, 42.7266541, -46.7792816, 27.5947819
1: -4.7651372, 17.5528717, -14.6245756, 49.7019196, -54.2854919, 32.1774483
2: -5.2724891, 17.0362396, -15.3455172, 48.6205978, -53.7413063, 32.3817558
3: -7.4683981, 18.4647770, -22.4691143, 52.3405228, -59.6541290, 40.9338913
4: -8.7734575, 16.2064075, -24.6704941, 46.9614105, -55.7348671, 40.8768997

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1669999, upper bound: 47.1636863
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1668913, upper bound: 47.1667419
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.2849174, 15.4173374, -12.3654547, 42.4875069, -46.6353416, 27.7827873
1: -4.8534427, 17.8700066, -14.5352459, 49.4262047, -54.1043510, 32.4052467
2: -5.3807850, 17.3343678, -15.2558937, 48.3464508, -53.5809212, 32.5902634
3: -7.5899143, 18.7914963, -22.3359680, 52.0523415, -59.4943352, 41.1274643
4: -8.9349260, 16.4891949, -24.5359573, 46.6917572, -55.6266785, 41.0251541

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1711260, upper bound: 47.1647119
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712601, upper bound: 47.1677862
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.7235136, 44.1252480, -3.6550412, 13.4881115, -26.2116203, 47.6404305
1: -15.1153173, 51.2670212, -4.1366725, 15.6353807, -30.7506962, 55.2359962
2: -15.6082811, 50.1621857, -4.6018124, 15.1535597, -30.7618370, 54.6207924
3: -23.0708389, 53.8880844, -6.5202684, 16.4253178, -39.4961548, 60.2512169
4: -24.9927807, 48.4072723, -7.7535706, 14.3413677, -39.3341484, 56.1437950

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1635591, upper bound: 47.1672898
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1645225, upper bound: 47.1688673
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.6422262, 47.0618324, -4.2697020, 15.4405231, -29.0827484, 51.2166176
1: -16.1728191, 54.7214432, -4.8553205, 17.8905773, -34.0633965, 59.4292679
2: -16.7433243, 53.4933968, -5.3622108, 17.3688622, -34.1121864, 58.7350616
3: -24.6783352, 57.5419312, -7.6058207, 18.8133278, -43.4916611, 65.0255280
4: -26.8195095, 51.6375656, -8.9210691, 16.5208302, -43.3403320, 60.5586357

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1613342, upper bound: 47.1680485
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1620409, upper bound: 47.1690823
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.7235136, 44.1252480, -12.1971588, 42.2329979, -54.8283920, 56.2188568
1: -15.1153173, 51.2670212, -14.3848553, 49.1392746, -64.0328293, 65.4526291
2: -15.6082811, 50.1621857, -15.0338392, 47.9842834, -63.4069138, 65.0231018
3: -23.0708389, 53.8880844, -22.0773525, 51.7345276, -74.5533752, 75.7196274
4: -24.9927807, 48.4072723, -24.2065601, 46.2641449, -71.1133652, 72.4748764

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1642854, upper bound: 47.1606934
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1644244, upper bound: 47.1619423
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.6422262, 47.0618324, -12.9824533, 44.7604942, -58.2955551, 59.9438820
1: -16.1728191, 54.7214432, -15.3028240, 52.0876389, -68.0679703, 69.8337097
2: -16.7433243, 53.4933968, -15.9770174, 50.8608208, -67.4350662, 69.3036194
3: -24.6783352, 57.5419312, -23.4612408, 54.8361893, -79.2984009, 80.7825851
4: -26.8195095, 51.6375656, -25.6882267, 49.0547791, -75.7574768, 77.2066345

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1621109, upper bound: 47.1607528
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1620017, upper bound: 47.1620017
time: 0.82 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.51 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 4, lower bound: -47.1797884, upper bound: 47.1797884
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 4, lower bound: -47.1797884, upper bound: 47.1799785
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 4, lower bound: -47.1673438, upper bound: 47.1715203
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 4, lower bound: -47.1712896, upper bound: 47.1712896
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 4, lower bound: -47.1669999, upper bound: 47.1636863
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 4, lower bound: -47.1668913, upper bound: 47.1667419
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 4, lower bound: -47.1711260, upper bound: 47.1647119
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 4, lower bound: -47.1712601, upper bound: 47.1677862
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 4, lower bound: -47.1635591, upper bound: 47.1672898
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 4, lower bound: -47.1645225, upper bound: 47.1688673
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 4, lower bound: -47.1613342, upper bound: 47.1680485
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 4, lower bound: -47.1620409, upper bound: 47.1690823
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 4, lower bound: -47.1642854, upper bound: 47.1606934
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 4, lower bound: -47.1644244, upper bound: 47.1619423
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 4, lower bound: -47.1621109, upper bound: 47.1607528
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 4, lower bound: -47.1620017, upper bound: 47.1620017

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.1957455, 15.1547375, -4.1957455, 15.1547375, -19.3504810, 19.3504829
1: -4.7651372, 17.5528717, -4.7651372, 17.5528717, -22.3180084, 22.3180084
2: -5.2724891, 17.0362396, -5.2724891, 17.0362396, -22.3087254, 22.3087254
3: -7.4683981, 18.4647770, -7.4683981, 18.4647770, -25.9331741, 25.9331741
4: -8.7734575, 16.2064075, -8.7734575, 16.2064075, -24.9798641, 24.9798641

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1704953, upper bound: 47.1668347
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658705, upper bound: 47.1658705
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.1957455, 15.1547375, -4.2849174, 15.4173374, -19.6130829, 19.4396477
1: -4.7651372, 17.5528717, -4.8534427, 17.8700066, -22.6351433, 22.4063129
2: -5.2724891, 17.0362396, -5.3807850, 17.3343678, -22.6068497, 22.4170246
3: -7.4683981, 18.4647770, -7.5899143, 18.7914963, -26.2598953, 26.0546913
4: -8.7734575, 16.2064075, -8.9349260, 16.4891949, -25.2626514, 25.1413326

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1704953, upper bound: 47.1710152
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658705, upper bound: 47.1702452
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.1808944, 15.0727634, -4.0143266, 14.6099663, -18.7908611, 19.0870895
1: -4.7290931, 17.4710522, -4.5484071, 16.9368134, -21.6659069, 22.0194588
2: -5.2520933, 16.9376259, -5.0499206, 16.4172859, -21.6693783, 21.9875450
3: -7.4023943, 18.3744202, -7.1442070, 17.8195877, -25.2219810, 25.5186272
4: -8.7406473, 16.0981236, -8.4604931, 15.5761414, -24.3167820, 24.5586166

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679817, upper bound: 47.1718355
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1685196, upper bound: 47.1717081
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.1182623, 14.8753653, -6.1268773, 21.1319256, -25.2501869, 21.0022411
1: -4.6474895, 17.2504826, -6.9795294, 24.5091591, -29.1566467, 24.2300072
2: -5.1824946, 16.7144413, -7.5739384, 23.8390598, -29.0215549, 24.2883797
3: -7.2841425, 18.1490345, -10.7911854, 25.7526836, -33.0368271, 28.9402199
4: -8.6388998, 15.8748178, -12.2735806, 22.8702431, -31.5091419, 28.1483994

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1669025, upper bound: 47.1712896
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1669025, upper bound: 47.1712896
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.0833750, 14.7841043, -12.1702709, 41.8467407, -45.7835121, 26.9543743
1: -4.6335025, 17.1229076, -14.2991581, 48.6807442, -53.1302567, 31.4220657
2: -5.1342368, 16.6117115, -15.0143929, 47.6059799, -52.5867386, 31.6260986
3: -7.2682552, 18.0172653, -21.9773674, 51.2706032, -58.3819008, 39.9946289
4: -8.5650139, 15.7886963, -24.1716404, 45.9593430, -54.5243568, 39.9603348

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1668913, upper bound: 47.1636863
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1668913, upper bound: 47.1636863
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.0468984, 14.6650085, -14.0867510, 47.7834320, -51.6523514, 28.7517586
1: -4.5774031, 16.9919128, -16.5063572, 55.5973778, -59.9505539, 33.4982605
2: -5.0941558, 16.4739304, -17.2990150, 54.3670197, -59.2731895, 33.7729454
3: -7.1891718, 17.8820801, -25.2855511, 58.5063286, -65.4967270, 43.1676178
4: -8.5069218, 15.6478243, -27.6406574, 52.5546074, -61.0387039, 43.2884827

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1668913, upper bound: 47.1667419
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1668913, upper bound: 47.1667419
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.1808944, 15.0727634, -12.0933552, 41.5997849, -45.6400795, 27.1661186
1: -4.7290931, 17.4710522, -14.2052231, 48.3959999, -52.9476318, 31.6762714
2: -5.2520933, 16.9376259, -14.9218884, 47.3218040, -52.4259491, 31.8595142
3: -7.4023943, 18.3744202, -21.8371353, 50.9724998, -58.2250328, 40.2115555
4: -8.7406473, 16.0981236, -24.0332489, 45.6784821, -54.4191284, 40.1313705

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1669159, upper bound: 47.1603345
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1675096, upper bound: 47.1603417
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.1182623, 14.8753653, -14.0114784, 47.5410194, -51.4867134, 28.8868427
1: -4.6474895, 17.2504826, -16.4144878, 55.3184624, -59.7467918, 33.6649666
2: -5.1824946, 16.7144413, -17.2081146, 54.0881157, -59.0870590, 33.9225540
3: -7.2841425, 18.1490345, -25.1474400, 58.2140923, -65.3054886, 43.2964745
4: -8.6388998, 15.8748178, -27.5049858, 52.2783699, -60.9023819, 43.3798027

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1668913, upper bound: 47.1677862
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712601, upper bound: 47.1677862
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -12.7021408, 44.0500183, -3.5186505, 12.9939127, -25.6960487, 47.4259186
1: -15.0893373, 51.1789589, -3.9814668, 15.0442829, -30.1336155, 54.9893303
2: -15.5827179, 50.0758247, -4.4331231, 14.5787764, -30.1614914, 54.3632240
3: -23.0318279, 53.7962608, -6.2797298, 15.8082066, -38.8400345, 59.9145432
4: -24.9526234, 48.3244438, -7.4775591, 13.8006105, -38.7532349, 55.7803192

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1552244, upper bound: 47.1644290
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1552244, upper bound: 47.1672898
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.6010370, 43.7221794, -3.5540607, 13.0886898, -25.6897278, 47.1382027
1: -14.9678202, 50.8000908, -4.0067978, 15.1727848, -30.1405964, 54.6400185
2: -15.4633322, 49.7017403, -4.4818072, 14.6908112, -30.1541443, 54.0419655
3: -22.8519020, 53.4007187, -6.3091884, 15.9443550, -38.7962570, 59.5542564
4: -24.7728081, 47.9589424, -7.5522461, 13.8996334, -38.6724396, 55.4962387

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1558857, upper bound: 47.1663374
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1558857, upper bound: 47.1663374
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.6197872, 46.9830093, -4.1364107, 14.9464703, -28.5662575, 51.0020485
1: -16.1451607, 54.6290588, -4.6979609, 17.3066120, -33.4517746, 59.1761169
2: -16.7164764, 53.4028320, -5.1991415, 16.8003349, -33.5168076, 58.4786606
3: -24.6368771, 57.4454765, -7.3638830, 18.2069435, -42.8438187, 64.6834488
4: -26.7774162, 51.5504646, -8.6525440, 15.9833679, -42.7607841, 60.2030067

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1631665
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630416
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.5257454, 46.6794815, -4.2387457, 15.2444744, -28.7702122, 50.8047447
1: -16.0332451, 54.2784653, -4.7994285, 17.6647930, -33.6980362, 58.9314842
2: -16.6045837, 53.0564537, -5.3204622, 17.1381111, -33.7426949, 58.2580986
3: -24.4712982, 57.0794525, -7.5042071, 18.5748806, -43.0461807, 64.4634552
4: -26.6087933, 51.2123032, -8.8327179, 16.3050137, -42.9138069, 60.0450211

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1675096
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1677092
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -12.5730391, 43.6339951, -11.8766890, 41.1671600, -53.6034927, 55.3995247
1: -14.9346256, 50.6961212, -13.9962606, 47.9045219, -62.6082420, 64.4869843
2: -15.4274101, 49.6002922, -14.6426229, 46.7595749, -61.9931641, 64.0631409
3: -22.8007374, 53.2917328, -21.4930058, 50.4418564, -72.9824066, 74.5346832
4: -24.7185841, 47.8567009, -23.6159935, 45.0614471, -69.6309509, 71.3290100

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1554959, upper bound: 47.1585571
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1638483, upper bound: 47.1601285
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.4961605, 43.3604507, -13.9816074, 47.7117386, -60.0188713, 57.2178802
1: -14.8327265, 50.3877983, -16.4229107, 55.5173035, -70.0428238, 66.6036224
2: -15.3373652, 49.2837448, -17.1371613, 54.1926689, -69.2679825, 66.2344742
3: -22.6511841, 52.9712524, -25.1252499, 58.4217987, -80.7406235, 77.8362885
4: -24.5848045, 47.5498123, -27.4321232, 52.2975426, -76.6868515, 74.8271027

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1478240, upper bound: 47.1453483
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1323162, upper bound: 47.1440695
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.4948425, 46.5756264, -12.6725626, 43.7337646, -57.1146355, 59.1410866
1: -15.9937048, 54.1570740, -14.9290380, 50.8957863, -66.6880341, 68.8899841
2: -16.5651340, 52.9350166, -15.5986948, 49.6811180, -66.0683823, 68.3602753
3: -24.4102631, 56.9524040, -22.8974323, 53.5891685, -77.7745438, 79.6247101
4: -26.5488319, 51.0896149, -25.1156292, 47.8959885, -74.3233261, 76.0813751

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0735859, upper bound: 47.1042684
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1590336, upper bound: 47.1575608
time: 1.30 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.4398212, 46.3873100, -14.8131046, 50.3851318, -63.6811218, 61.0847664
1: -15.9226055, 53.9443970, -17.3976555, 58.6404915, -74.3202438, 71.1494522
2: -16.5019360, 52.7186089, -18.1346149, 57.2443695, -73.5297928, 70.6791534
3: -24.3067513, 56.7319603, -26.5943394, 61.7074814, -85.7445221, 83.0946045
4: -26.4559593, 50.8813438, -28.9921055, 55.2607880, -81.5691299, 79.7405624

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0714227, upper bound: 47.1005570
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0714227, upper bound: 47.1589556
time: 1.14 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.41 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1704953, upper bound: 47.1668347
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1658705, upper bound: 47.1658705
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1704953, upper bound: 47.1710152
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1658705, upper bound: 47.1702452
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1679817, upper bound: 47.1718355
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1685196, upper bound: 47.1717081
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1669025, upper bound: 47.1712896
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1669025, upper bound: 47.1712896
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1668913, upper bound: 47.1636863
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1668913, upper bound: 47.1636863
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1668913, upper bound: 47.1667419
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1668913, upper bound: 47.1667419
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1669159, upper bound: 47.1603345
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1675096, upper bound: 47.1603417
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1668913, upper bound: 47.1677862
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1712601, upper bound: 47.1677862
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1552244, upper bound: 47.1644290
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1552244, upper bound: 47.1672898
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1558857, upper bound: 47.1663374
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1558857, upper bound: 47.1663374
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1631665
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630416
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1675096
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1677092
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1554959, upper bound: 47.1585571
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1638483, upper bound: 47.1601285
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1478240, upper bound: 47.1453483
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1323162, upper bound: 47.1440695
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.0735859, upper bound: 47.1042684
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.1590336, upper bound: 47.1575608
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.0714227, upper bound: 47.1005570
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -47.0714227, upper bound: 47.1589556

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.9600012, 14.3796635, -4.0833750, 14.7841043, -18.7441025, 18.4630375
1: -4.4894509, 16.6531467, -4.6335025, 17.1229076, -21.6123581, 21.2866421
2: -4.9817009, 16.1492329, -5.1342368, 16.6117115, -21.5934124, 21.2834682
3: -7.0482631, 17.5268631, -7.2682552, 18.0172653, -25.0655251, 24.7951183
4: -8.3357964, 15.3324518, -8.5650139, 15.7886963, -24.1244926, 23.8974648

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658705, upper bound: 47.1658705
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658705, upper bound: 47.1658705
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.0743117, 20.9155369, -4.0468984, 14.6650085, -20.7393208, 24.9624348
1: -6.9165897, 24.2520409, -4.5774031, 16.9919128, -23.9085007, 28.8294430
2: -7.5073729, 23.5865192, -5.0941558, 16.4739304, -23.9813042, 28.6806755
3: -10.6918325, 25.4853954, -7.1891718, 17.8820801, -28.5739117, 32.6745682
4: -12.1613083, 22.6388607, -8.5069218, 15.6478243, -27.8091316, 31.1457787

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658705, upper bound: 47.1658705
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658705, upper bound: 47.1658705
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.9600012, 14.3796635, -4.1808944, 15.0727634, -19.0327625, 18.5605583
1: -4.4894509, 16.6531467, -4.7290931, 17.4710522, -21.9605026, 21.3822403
2: -4.9817009, 16.1492329, -5.2520933, 16.9376259, -21.9193268, 21.4013214
3: -7.0482631, 17.5268631, -7.4023943, 18.3744202, -25.4226780, 24.9292564
4: -8.3357964, 15.3324518, -8.7406473, 16.0981236, -24.4339199, 24.0730991

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1669025, upper bound: 47.1702452
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1669025, upper bound: 47.1702452
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.0743117, 20.9155369, -4.1182623, 14.8753653, -20.9496765, 25.0337963
1: -6.9165897, 24.2520409, -4.6474895, 17.2504826, -24.1670685, 28.8995266
2: -7.5073729, 23.5865192, -5.1824946, 16.7144413, -24.2218132, 28.7690144
3: -10.6918325, 25.4853954, -7.2841425, 18.1490345, -28.8408642, 32.7695389
4: -12.1613083, 22.6388607, -8.6388998, 15.8748178, -28.0361252, 31.2777596

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1669025, upper bound: 47.1702452
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1669025, upper bound: 47.1702452
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -3.3551190, 12.4388256, -3.3623033, 12.5284424, -15.8835611, 15.8011284
1: -3.8385346, 14.3716965, -3.7937446, 14.5262384, -18.3647728, 18.1654396
2: -4.2237635, 13.9857130, -4.2436256, 14.0566835, -18.2804470, 18.2293377
3: -6.0357733, 15.0850449, -6.0006800, 15.2669697, -21.3027420, 21.0857239
4: -7.0967364, 13.2603350, -7.2144132, 13.2603455, -20.3570824, 20.4747467

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1662872, upper bound: 47.1661842
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1655988, upper bound: 47.1696898
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.0713782, 14.6621799, -3.9623661, 14.4254932, -18.4968719, 18.6245422
1: -4.6038389, 16.9836750, -4.4894609, 16.7182636, -21.3221016, 21.4731350
2: -5.1088467, 16.4716721, -4.9845963, 16.2084904, -21.3173370, 21.4562683
3: -7.1989069, 17.8603210, -7.0516710, 17.5891666, -24.7880745, 24.9119911
4: -8.4979849, 15.6607428, -8.3516083, 15.3785486, -23.8765335, 24.0123463

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1677704, upper bound: 47.1707670
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1677704, upper bound: 47.1717081
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.0653152, 14.6894693, -6.1268773, 21.1319256, -25.1972408, 20.8163471
1: -4.5913391, 17.0269394, -6.9795294, 24.5091591, -29.1004982, 24.0064659
2: -5.1094751, 16.4968529, -7.5739384, 23.8390598, -28.9485359, 24.0707912
3: -7.1942153, 17.9111671, -10.7911854, 25.7526836, -32.9468994, 28.7023525
4: -8.5246019, 15.6643715, -12.2735806, 22.8702431, -31.3948441, 27.9379520

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679025, upper bound: 47.1673740
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678801, upper bound: 47.1678801
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.1746607, 21.1083221, -6.1268773, 21.1319256, -27.3065872, 27.2351952
1: -7.0086069, 24.4902382, -6.9795294, 24.5091591, -31.5177650, 31.4697666
2: -7.6236777, 23.7928734, -7.5739384, 23.8390598, -31.4627304, 31.3668118
3: -10.8146791, 25.7496147, -10.7911854, 25.7526836, -36.5673637, 36.5408020
4: -12.3101854, 22.8557110, -12.2735806, 22.8702431, -35.1804276, 35.1292915

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679025, upper bound: 47.1672810
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678801, upper bound: 47.1678801
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.9600012, 14.3796635, -12.1702709, 41.8467407, -45.6600838, 26.5499344
1: -4.4894509, 16.6531467, -14.2991581, 48.6807442, -52.9867210, 30.9523029
2: -4.9817009, 16.1492329, -15.0143929, 47.6059799, -52.4350357, 31.1636219
3: -7.0482631, 17.5268631, -21.9773674, 51.2706032, -58.1627846, 39.5042267
4: -8.3357964, 15.3324518, -24.1716404, 45.9593430, -54.2951393, 39.5040855

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1661898, upper bound: 47.1635280
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1661898, upper bound: 47.1636863
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.0531664, 20.8245430, -12.1702709, 41.8467407, -47.7465744, 32.9948120
1: -6.8937969, 24.1513729, -14.2991581, 48.6807442, -55.3789330, 38.4505272
2: -7.4937367, 23.4771976, -15.0143929, 47.6059799, -54.9473343, 38.4915924
3: -10.6671381, 25.4023323, -21.9773674, 51.2706032, -61.7748413, 47.3796921
4: -12.1349783, 22.5499516, -24.1716404, 45.9593430, -58.0841293, 46.7215919

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1661898, upper bound: 47.1635280
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1661898, upper bound: 47.1636863
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.9463775, 14.3346872, -14.0867510, 47.7834320, -51.5522079, 28.4214382
1: -4.4730644, 16.6013870, -16.5063572, 55.5973778, -59.8478203, 33.1077385
2: -4.9651532, 16.0977135, -17.2990150, 54.3670197, -59.1450882, 33.3967285
3: -7.0233083, 17.4727135, -25.2855511, 58.5063286, -65.3319092, 42.7582626
4: -8.3106852, 15.2818718, -27.6406574, 52.5546074, -60.8434982, 42.9225273

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658938, upper bound: 47.1641494
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658938, upper bound: 47.1667419
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.0531664, 20.8245430, -14.0867510, 47.7834320, -53.6523590, 34.9112892
1: -6.8937969, 24.1513729, -16.5063572, 55.5973778, -62.2564240, 40.6577187
2: -7.4937367, 23.4771976, -17.2990150, 54.3670197, -61.6738701, 40.7762146
3: -10.6671381, 25.4023323, -25.2855511, 58.5063286, -68.9689178, 50.6878738
4: -12.1349783, 22.5499516, -27.6406574, 52.5546074, -64.6522369, 50.1906090

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

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
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658938, upper bound: 47.1639770
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658938, upper bound: 47.1644260
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -3.3551190, 12.4388256, -11.3051281, 39.0507355, -42.2165833, 23.7439537
1: -3.8385346, 14.3716965, -13.2842922, 45.4222755, -49.0340767, 27.6559887
2: -4.2237635, 13.9857130, -13.9681559, 44.4126167, -48.4473610, 27.9538689
3: -6.0357733, 15.0850449, -20.4431820, 47.8500099, -53.6832199, 35.5282288
4: -7.0967364, 13.2603350, -22.5378265, 42.8531494, -49.9057083, 35.7981606

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1651694, upper bound: 47.1603345
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1651694, upper bound: 47.1603345
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.0713782, 14.6621799, -12.0209131, 41.3532181, -45.2871513, 26.6830921
1: -4.6038389, 16.9836750, -14.1196585, 48.1078720, -52.5393181, 31.1033325
2: -5.1088467, 16.4716721, -14.8319740, 47.0399017, -52.0097694, 31.3036461
3: -7.1989069, 17.8603210, -21.7056675, 50.6695099, -57.7302856, 39.5659866
4: -8.4979849, 15.6607428, -23.8911266, 45.4054565, -53.9034424, 39.5518608

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660071, upper bound: 47.1603417
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1660071, upper bound: 47.1603417
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.0653152, 14.6894693, -14.0114784, 47.5410194, -51.4337196, 28.7009468
1: -4.5913391, 17.0269394, -16.4144878, 55.3184624, -59.6927299, 33.4414215
2: -5.1094751, 16.4968529, -17.2081146, 54.0881157, -59.0150452, 33.7049675
3: -7.1942153, 17.9111671, -25.1474400, 58.2140923, -65.2168655, 43.0586014
4: -8.5246019, 15.6643715, -27.5049858, 52.2783699, -60.7898026, 43.1693573

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1697134, upper bound: 47.1651795
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1697134, upper bound: 47.1677862
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.1746607, 21.1083221, -14.0114784, 47.5410194, -53.5292397, 35.1197929
1: -7.0086069, 24.4902382, -16.4144878, 55.3184624, -62.0915794, 40.9047241
2: -7.6236777, 23.7928734, -17.2081146, 54.0881157, -61.5297508, 41.0009804
3: -10.8146791, 25.7496147, -25.1474400, 58.2140923, -68.8291626, 50.8970566
4: -12.3101854, 22.8557110, -27.5049858, 52.2783699, -64.5581284, 50.3606949

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1697134, upper bound: 47.1651212
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1697134, upper bound: 47.1673725
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.4524336, 43.1678200, -3.5186505, 12.9939127, -25.4463463, 46.5333939
1: -14.7858372, 50.1457520, -3.9814668, 15.0442829, -29.8301201, 53.9432259
2: -15.2835922, 49.0636292, -4.4331231, 14.5787764, -29.8623695, 53.3407288
3: -22.5760632, 52.7194977, -6.2797298, 15.8082066, -38.3842545, 58.8273430
4: -24.4824944, 47.3548775, -7.4775591, 13.8006105, -38.2831001, 54.8038216

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1525606, upper bound: 47.1567351
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1523973, upper bound: 47.1627213
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.5043392, 43.3323669, -3.5186505, 12.9939127, -25.4982452, 46.7059364
1: -14.8310118, 50.3433380, -3.9814668, 15.0442829, -29.8752937, 54.1524048
2: -15.3490219, 49.2404137, -4.4331231, 14.5787764, -29.9277992, 53.5273438
3: -22.6377640, 52.9216881, -6.2797298, 15.8082066, -38.4459648, 59.0423775
4: -24.5850430, 47.5082436, -7.4775591, 13.8006105, -38.3856544, 54.9646263

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1525606, upper bound: 47.1588046
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1523973, upper bound: 47.1647500
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -12.4524336, 43.1678200, -3.5540607, 13.0886898, -25.5411224, 46.5711060
1: -14.7858372, 50.1457520, -4.0067978, 15.1727848, -29.9586201, 53.9693222
2: -15.2835922, 49.0636292, -4.4818072, 14.6908112, -29.9744034, 53.3909073
3: -22.5760632, 52.7194977, -6.3091884, 15.9443550, -38.5204163, 58.8591118
4: -24.4824944, 47.3548775, -7.5522461, 13.8996334, -38.3821182, 54.8825417

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1525606, upper bound: 47.1608397
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1523973, upper bound: 47.1647414
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.5043392, 43.3323669, -3.5540607, 13.0886898, -25.5930252, 46.7472038
1: -14.8310118, 50.3433380, -4.0067978, 15.1727848, -30.0037937, 54.1831741
2: -15.3490219, 49.2404137, -4.4818072, 14.6908112, -30.0398331, 53.5814590
3: -22.6377640, 52.9216881, -6.3091884, 15.9443550, -38.5821190, 59.0786858
4: -24.5850430, 47.5082436, -7.5522461, 13.8996334, -38.4846764, 55.0468025

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1525606, upper bound: 47.1622835
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1523973, upper bound: 47.1663575
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.3127680, 45.9683151, -4.0251579, 14.5801859, -27.8929520, 49.8715019
1: -15.7727365, 53.4514465, -4.5676088, 16.8815708, -32.6543083, 57.8620110
2: -16.3455467, 52.2382240, -5.0616937, 16.3810062, -32.7265549, 57.1703491
3: -24.0788822, 56.2157745, -7.1652679, 17.7635612, -41.8424377, 63.2479935
4: -26.2137661, 50.4090462, -8.4455528, 15.5700531, -41.7838211, 58.8545952

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1630416
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1630416
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -15.4723587, 52.6685753, -3.9875202, 14.4585590, -29.9309158, 56.5024147
1: -18.2672024, 61.2539215, -4.5099778, 16.7477379, -35.0149384, 65.5640106
2: -18.8973389, 59.8631897, -5.0204396, 16.2402802, -35.1376190, 64.7159348
3: -27.8071709, 64.3895111, -7.0842490, 17.6255322, -45.4326859, 71.2977371
4: -30.1080036, 57.8368988, -8.3858852, 15.4260864, -45.5340881, 66.2116165

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1437576, upper bound: 47.1608537
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630416
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630416
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.2168636, 45.6590157, -4.1345463, 14.8992052, -28.1160660, 49.6754074
1: -15.6575136, 53.0947304, -4.6755261, 17.2650185, -32.9225235, 57.6176338
2: -16.2312317, 51.8845634, -5.1914325, 16.7405243, -32.9717560, 56.9509468
3: -23.9082832, 55.8423767, -7.3163252, 18.1568966, -42.0651779, 63.0314522
4: -26.0420685, 50.0627861, -8.6377726, 15.9132681, -41.9553375, 58.7005577

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1675096
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1675096
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -15.3765392, 52.3601875, -4.0726924, 14.7063904, -30.0829296, 56.2831345
1: -18.1519470, 60.8987846, -4.5939703, 17.0499229, -35.2018700, 65.2974854
2: -18.7832756, 59.5100441, -5.1229863, 16.5228863, -35.3061600, 64.4695587
3: -27.6355209, 64.0175629, -7.1996875, 17.9365273, -45.5720444, 71.0467758
4: -29.9361267, 57.4898186, -8.5380516, 15.6945467, -45.6306725, 66.0247726

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1593617, upper bound: 47.1609134
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1677092
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1677092
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.3055563, 42.6883698, -11.8585997, 41.1026230, -53.2662239, 54.4225845
1: -14.6106949, 49.5879669, -13.9744129, 47.8285370, -62.2046700, 63.3407326
2: -15.1071949, 48.5159569, -14.6207066, 46.6853981, -61.5934753, 62.9438210
3: -22.3145161, 52.1369019, -21.4596252, 50.3625450, -72.4129333, 73.3329697
4: -24.2144508, 46.8198509, -23.5810757, 44.9905281, -69.0510254, 70.2480698

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
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
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1547435, upper bound: 47.1585571
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1547435, upper bound: 47.1585571
time: 0.50 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.3575659, 42.8503761, -11.7671986, 40.8060455, -53.0193405, 54.5029106
1: -14.6542559, 49.7838936, -13.8649282, 47.4857330, -61.9093857, 63.4430313
2: -15.1727819, 48.6880531, -14.5117846, 46.3469391, -61.3225937, 63.0200844
3: -22.3727036, 52.3375664, -21.2973042, 50.0047188, -72.1173096, 73.3885498
4: -24.3176918, 46.9669991, -23.4166698, 44.6599350, -68.8289337, 70.2410202

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1318277, upper bound: 47.1556189
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1322534, upper bound: 47.1505591
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.4512825, 29.9177780, -11.8500280, 40.9499054, -49.1415520, 41.5593796
1: -10.0009956, 34.7798309, -13.9092398, 47.7166672, -57.3661232, 48.4144249
2: -10.4832430, 33.9773903, -14.6002903, 46.5016174, -56.6659126, 48.3276863
3: -15.4637012, 36.6414032, -21.4004955, 50.2560883, -65.3439941, 57.7316628
4: -17.1012173, 32.6917076, -23.6179562, 44.7642479, -61.6347656, 56.1172829

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1465659, upper bound: 47.1447311
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1469569, upper bound: 47.1441948
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.0940142, 42.0935440, -13.7710209, 47.0546379, -58.8784866, 55.6145325
1: -14.3560333, 48.9226570, -16.1752529, 54.7553101, -68.6952209, 64.7322159
2: -14.8593388, 47.8379364, -16.8893147, 53.4441071, -67.9377213, 64.3917465
3: -21.9431038, 51.4284019, -24.7579193, 57.6189995, -79.1112366, 75.7639847
4: -23.8534622, 46.1411400, -27.0517311, 51.5683823, -75.1032867, 72.9034958

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1323162, upper bound: 47.1440695
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1323162, upper bound: 47.1440695
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.4911661, 46.5648232, -12.3878098, 42.7886238, -56.1541824, 58.8327904
1: -16.0070782, 54.1065292, -14.5817633, 49.7867661, -65.5855179, 68.4791489
2: -16.5108910, 52.9107437, -15.2386475, 48.5907440, -64.9143524, 67.9625168
3: -24.3775063, 56.8119125, -22.3615017, 52.3874931, -76.5320587, 78.9353333
4: -26.3508663, 51.0408745, -24.5067291, 46.8149147, -73.0376740, 75.4071045

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
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
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0607203, upper bound: 47.0673466
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0607203, upper bound: 47.1042684
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.1343088, 45.3914719, -12.5070200, 43.1937141, -56.2051773, 57.7813377
1: -15.5655718, 52.7887650, -14.7317982, 50.2726364, -65.6313248, 67.3150635
2: -16.1361809, 51.5891380, -15.3994274, 49.0639000, -65.0121155, 66.8004150
3: -23.7680836, 55.5371132, -22.6020145, 52.9429016, -76.4789276, 77.9008255
4: -25.9063053, 49.7737732, -24.8204765, 47.2892303, -73.0674667, 74.4598083

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1419805, upper bound: 47.1568351
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0735859, upper bound: 47.1532043
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.4324923, 46.3676682, -14.5181370, 49.4040718, -62.6813965, 60.7545433
1: -15.9302301, 53.8837090, -17.0293884, 57.4898987, -73.1704788, 70.7055969
2: -16.4444027, 52.6847839, -17.7583504, 56.1096878, -72.3281174, 70.2551956
3: -24.2650185, 56.5806656, -26.0261631, 60.4575310, -84.4448090, 82.3617630
4: -26.2521286, 50.8222694, -28.3608913, 54.1311913, -80.2290192, 79.0333405

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0582481, upper bound: 47.0588107
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0681180, upper bound: 47.0987187
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0688270, upper bound: 47.0983559
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.0819521, 45.2101440, -14.6486320, 49.8555489, -62.7840385, 59.7361374
1: -15.4975967, 52.5813141, -17.2056541, 58.0295486, -73.2787399, 69.5839233
2: -16.0759850, 51.3785324, -17.9414921, 56.6397018, -72.4902954, 69.1330795
3: -23.6696301, 55.3256111, -26.3059330, 61.0756607, -84.4694595, 81.3870621
4: -25.8180523, 49.5721169, -28.7044353, 54.6672134, -80.3316879, 78.1341629

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1589124, upper bound: 47.1589556
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1589124, upper bound: 47.1589556
time: 0.60 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.40 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1658705, upper bound: 47.1658705
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1658705, upper bound: 47.1658705
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1658705, upper bound: 47.1658705
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1658705, upper bound: 47.1658705
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1669025, upper bound: 47.1702452
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1669025, upper bound: 47.1702452
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1669025, upper bound: 47.1702452
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1669025, upper bound: 47.1702452
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1662872, upper bound: 47.1661842
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1655988, upper bound: 47.1696898
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1677704, upper bound: 47.1707670
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1677704, upper bound: 47.1717081
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1679025, upper bound: 47.1673740
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1678801, upper bound: 47.1678801
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1679025, upper bound: 47.1672810
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1678801, upper bound: 47.1678801
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1661898, upper bound: 47.1635280
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1661898, upper bound: 47.1636863
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1661898, upper bound: 47.1635280
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1661898, upper bound: 47.1636863
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1658938, upper bound: 47.1641494
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1658938, upper bound: 47.1667419
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1658938, upper bound: 47.1639770
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1658938, upper bound: 47.1644260
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1651694, upper bound: 47.1603345
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1651694, upper bound: 47.1603345
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1660071, upper bound: 47.1603417
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1660071, upper bound: 47.1603417
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1697134, upper bound: 47.1651795
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1697134, upper bound: 47.1677862
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1697134, upper bound: 47.1651212
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1697134, upper bound: 47.1673725
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1525606, upper bound: 47.1567351
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1523973, upper bound: 47.1627213
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1525606, upper bound: 47.1588046
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1523973, upper bound: 47.1647500
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1525606, upper bound: 47.1608397
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1523973, upper bound: 47.1647414
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1525606, upper bound: 47.1622835
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1523973, upper bound: 47.1663575
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1630416
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1630416
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630416
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630416
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1675096
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1675096
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1677092
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1677092
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1547435, upper bound: 47.1585571
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1547435, upper bound: 47.1585571
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1318277, upper bound: 47.1556189
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1322534, upper bound: 47.1505591
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1465659, upper bound: 47.1447311
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1469569, upper bound: 47.1441948
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1323162, upper bound: 47.1440695
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1323162, upper bound: 47.1440695
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.0607203, upper bound: 47.0673466
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.0607203, upper bound: 47.1042684
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1419805, upper bound: 47.1568351
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.0735859, upper bound: 47.1532043
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.0681180, upper bound: 47.0987187
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.0688270, upper bound: 47.0983559
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1589124, upper bound: 47.1589556
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.40
Output dim: 4, lower bound: -47.1589124, upper bound: 47.1589556

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.9600012, 14.3796635, -3.9600012, 14.3796635, -18.3396645, 18.3396645
1: -4.4894509, 16.6531467, -4.4894509, 16.6531467, -21.1425972, 21.1425972
2: -4.9817009, 16.1492329, -4.9817009, 16.1492329, -21.1309319, 21.1309319
3: -7.0482631, 17.5268631, -7.0482631, 17.5268631, -24.5751228, 24.5751228
4: -8.3357964, 15.3324518, -8.3357964, 15.3324518, -23.6682472, 23.6682472

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1641622, upper bound: 47.1646429
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1690702, upper bound: 47.1645599
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.9600012, 14.3796635, -6.0531664, 20.8245430, -24.7845440, 20.4328308
1: -4.4894509, 16.6531467, -6.8937969, 24.1513729, -28.6408215, 23.5469360
2: -4.9817009, 16.1492329, -7.4937367, 23.4771976, -28.4588985, 23.6429672
3: -7.0482631, 17.5268631, -10.6671381, 25.4023323, -32.4505844, 28.1940002
4: -8.3357964, 15.3324518, -12.1349783, 22.5499516, -30.8857460, 27.4674301

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1641622, upper bound: 47.1646429
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1641622, upper bound: 47.1645599
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.0743117, 20.9155369, -3.9463775, 14.3346872, -20.4089985, 24.8619137
1: -6.9165897, 24.2520409, -4.4730644, 16.6013870, -23.5179729, 28.7251015
2: -7.5073729, 23.5865192, -4.9651532, 16.0977135, -23.6050854, 28.5516720
3: -10.6918325, 25.4853954, -7.0233083, 17.4727135, -28.1645451, 32.5087051
4: -12.1613083, 22.6388607, -8.3106852, 15.2818718, -27.4431801, 30.9495430

Time for backsubstitution: 1.77 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0416667, mid=0.0416667, abs_max=50.8192024230957
rel_dist={4: [-47.180735877080295, 47.18073587708028]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1132.18 seconds
