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
execution time: IAR + LP analysis = 1.69 + 1.94 = 3.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -47.1809221, upper bound: 47.1809221


# Binary Search by BASE starts (time budget: 1196.38 seconds, max iter: 100)

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


# Relational Split (RS_dual_Z) starts
Time budget: 1132.42 seconds

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.22 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.22
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.22
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.50 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.47 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.64 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.64
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.64
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.64
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.64
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.76 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.22 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.22
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.22
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.22
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.22
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.22
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.22
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.22
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.22
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
Binary search (step 0): status=Status.VERIFIED, low=0.1666667, high=0.3333333, mid=0.1666667, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.24 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.49 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.51 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.74 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.50 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.93 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.93
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.93
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.93
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.93
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.93
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.93
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.93
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.93
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
Binary search (step 1): status=Status.VERIFIED, low=0.2500000, high=0.3333333, mid=0.2500000, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 2) starts
Candidate diff: 0.2916666


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.49 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.15 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.15
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.15
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.52 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.69 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.20 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.64 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.92 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.92
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.92
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.92
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.92
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.92
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.92
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.92
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.92
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
Binary search (step 2): status=Status.VERIFIED, low=0.2916666, high=0.3333333, mid=0.2916666, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 3) starts
Candidate diff: 0.3125000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.47 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.13 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.13
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.13
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.52 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.54 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.56 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.89 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.89
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.89
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.89
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.89
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.89
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.89
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.89
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.89
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
Binary search (step 3): status=Status.VERIFIED, low=0.3125000, high=0.3333333, mid=0.3125000, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 4) starts
Candidate diff: 0.3229166


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.48 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.13 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.13
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.13
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.65 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.46 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.50 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.80 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.80
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.80
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.80
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.80
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.80
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.80
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.80
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.80
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
Binary search (step 4): status=Status.VERIFIED, low=0.3229166, high=0.3333333, mid=0.3229166, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 5) starts
Candidate diff: 0.3281250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.48 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.14 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.14
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.14
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.54 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.53 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.33 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.33
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.33
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.33
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.33
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.55 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.82 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
Binary search (step 5): status=Status.VERIFIED, low=0.3281250, high=0.3333333, mid=0.3281250, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 6) starts
Candidate diff: 0.3307291


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.49 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.17 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.17
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.17
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.75 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.20 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.56 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.84 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.84
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.84
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.84
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.84
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.84
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.84
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.84
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.84
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
Binary search (step 6): status=Status.VERIFIED, low=0.3307291, high=0.3333333, mid=0.3307291, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 7) starts
Candidate diff: 0.3320312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.50 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.16 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.16
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.16
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.51 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.53 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.53 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.81 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
Binary search (step 7): status=Status.VERIFIED, low=0.3320312, high=0.3333333, mid=0.3320312, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 8) starts
Candidate diff: 0.3326823


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.47 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.13 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.13
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.13
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.49 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.52 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.42 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.42
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.42
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.42
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.42
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.54 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.76 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
Binary search (step 8): status=Status.VERIFIED, low=0.3326823, high=0.3333333, mid=0.3326823, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 9) starts
Candidate diff: 0.3330078


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.48 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.16 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.16
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.16
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.52 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.51 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.39 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.39
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.39
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.39
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.39
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.54 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.90 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.90
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.90
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.90
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.90
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.90
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.90
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.90
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.90
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
Binary search (step 9): status=Status.VERIFIED, low=0.3330078, high=0.3333333, mid=0.3330078, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 10) starts
Candidate diff: 0.3331706


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.48 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.16 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.16
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.16
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.46 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.52 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.30 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.53 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.76 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
Binary search (step 10): status=Status.VERIFIED, low=0.3331706, high=0.3333333, mid=0.3331706, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 11) starts
Candidate diff: 0.3332519


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.47 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.10 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.75 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.79 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.35 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.56 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.76 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
Binary search (step 11): status=Status.VERIFIED, low=0.3332519, high=0.3333333, mid=0.3332519, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 12) starts
Candidate diff: 0.3332926


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.47 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.08 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.08
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.08
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.53 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.50 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.86 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.54 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.77 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
Binary search (step 12): status=Status.VERIFIED, low=0.3332926, high=0.3333333, mid=0.3332926, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 13) starts
Candidate diff: 0.3333130


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.47 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.11 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.11
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.11
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.56 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.55 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.93 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.56 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.76 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.76
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
Binary search (step 13): status=Status.VERIFIED, low=0.3333130, high=0.3333333, mid=0.3333130, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 14) starts
Candidate diff: 0.3333231


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.48 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.12 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.12
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.12
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.56 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.53 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.18 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.54 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.85 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.85
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.85
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.85
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.85
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.85
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.85
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.85
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.85
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
Binary search (step 14): status=Status.VERIFIED, low=0.3333231, high=0.3333333, mid=0.3333231, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 15) starts
Candidate diff: 0.3333282


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.48 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.13 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.13
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.13
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.56 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.59 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.50 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.69 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.69
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.69
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.69
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.69
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.69
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.69
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.69
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.69
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
Binary search (step 15): status=Status.VERIFIED, low=0.3333282, high=0.3333333, mid=0.3333282, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 16) starts
Candidate diff: 0.3333308


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.47 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.11 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.11
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.11
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.53 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.52 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.30 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.49 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.72 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.72
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.72
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.72
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.72
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.72
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.72
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.72
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.72
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
Binary search (step 16): status=Status.VERIFIED, low=0.3333308, high=0.3333333, mid=0.3333308, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 17) starts
Candidate diff: 0.3333320


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.48 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.12 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.12
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.12
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.49 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.51 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.22 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.54 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.75 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.75
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.75
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.75
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.75
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.75
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.75
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.75
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.75
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
Binary search (step 17): status=Status.VERIFIED, low=0.3333320, high=0.3333333, mid=0.3333320, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary search (step 18) starts
Candidate diff: 0.3333327


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.47 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.09 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.09
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.09
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.50 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
time: 0.53 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1619718
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 4, lower bound: -47.1619718, upper bound: 47.1622348

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003
1: -10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643
2: -10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861
3: -15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287
4: -17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
time: 0.54 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.88 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.88
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.88
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.88
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.88
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.88
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.88
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.88
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.88
Output dim: 4, lower bound: -47.0392774, upper bound: 47.0392774
Binary search (step 18): status=Status.VERIFIED, low=0.3333327, high=0.3333333, mid=0.3333327, abs_max=50.8192024230957
rel_dist={4: [-47.1809221486041, 47.1809221486041]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.3333326776822787
execution time: 433.52 seconds
