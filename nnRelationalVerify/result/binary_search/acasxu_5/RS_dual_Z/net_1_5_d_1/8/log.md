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
execution time: IAR + LP analysis = 2.49 + 1.96 = 4.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -27.5477966, upper bound: 27.5477966


# Binary Search by BASE starts (time budget: 1195.55 seconds, max iter: 100)

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
Binary search time: 71.94 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1123.62 seconds

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940
time: 0.77 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.78 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.78
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.78
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5431460, upper bound: 27.5445494
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5432626, upper bound: 27.5431460
time: 0.69 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5431460, upper bound: 27.5460723
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5431460, upper bound: 27.5432626
time: 0.81 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.46 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.46
Output dim: 3, lower bound: -27.5431460, upper bound: 27.5445494
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.46
Output dim: 3, lower bound: -27.5432626, upper bound: 27.5431460
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.46
Output dim: 3, lower bound: -27.5431460, upper bound: 27.5460723
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.46
Output dim: 3, lower bound: -27.5431460, upper bound: 27.5432626

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5176531, upper bound: 27.5167048
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
time: 0.72 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.06 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 4.06
Output dim: 3, lower bound: -27.5176531, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
Binary search (step 0): status=Status.VERIFIED, low=0.0625000, high=0.1250000, mid=0.0625000, abs_max=31.572500228881836
rel_dist={3: [-27.547796646245764, 27.54779664624577]}

## Binary search (step 1) starts
Candidate diff: 0.0937500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940
time: 0.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5454382
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5456764
time: 0.73 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456010
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5472606
time: 0.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.09
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5454382
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.09
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5456764
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.09
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456010
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.09
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5472606

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5438520
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5455850, upper bound: 27.5428002
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5441190
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5427901
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428747
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5455850
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428990
time: 0.80 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.08 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5438520
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 3, lower bound: -27.5455850, upper bound: 27.5428002
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5441190
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5427901
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428747
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5455850
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428990

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5397462
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5345851
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5388474
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5395017, upper bound: 27.5344886
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5418390
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5411491
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5391825
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5397372, upper bound: 27.5377179
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5377179, upper bound: 27.5397372
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5391825, upper bound: 27.5344487
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5387729
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5418390, upper bound: 27.5343567
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5438503
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5436355
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5395017
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5378548
time: 0.84 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5397462
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5345851
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5388474
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 3, lower bound: -27.5395017, upper bound: 27.5344886
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5418390
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5411491
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5391825
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 3, lower bound: -27.5397372, upper bound: 27.5377179
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 3, lower bound: -27.5377179, upper bound: 27.5397372
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 3, lower bound: -27.5391825, upper bound: 27.5344487
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5387729
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 3, lower bound: -27.5418390, upper bound: 27.5343567
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5438503
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5436355
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5395017
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5378548

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5169085, upper bound: 27.5167048
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5169085, upper bound: 27.5167048
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5172478, upper bound: 27.5167048
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5176531, upper bound: 27.5167048
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5176531, upper bound: 27.5167048
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169915
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168325
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168603
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168352
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.78 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5169085, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5169085, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5172478, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5176531, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5176531, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169915
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168325
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168603
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168352
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
Binary search (step 1): status=Status.VERIFIED, low=0.0937500, high=0.1250000, mid=0.0937500, abs_max=31.572500228881836
rel_dist={3: [-27.547796646245764, 27.54779664624577]}

## Binary search (step 2) starts
Candidate diff: 0.1093750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940
time: 0.70 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5454382
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456764
time: 0.75 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5456764, upper bound: 27.5456010
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5472606
time: 0.55 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.95 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.95
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5454382
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.95
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456764
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.95
Output dim: 3, lower bound: -27.5456764, upper bound: 27.5456010
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.95
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5472606

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5438520
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5441190, upper bound: 27.5428002
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5441190
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5438520, upper bound: 27.5427901
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428747
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428990
time: 0.77 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.04 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5438520
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 3, lower bound: -27.5441190, upper bound: 27.5428002
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5441190
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 3, lower bound: -27.5438520, upper bound: 27.5427901
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428747
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428990

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5397462
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5395017, upper bound: 27.5345851
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5388474
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5438503, upper bound: 27.5344886
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5418390
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5411491
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5391825
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5397372, upper bound: 27.5377179
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5397372
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5391825, upper bound: 27.5344487
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5411491, upper bound: 27.5387729
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5418390, upper bound: 27.5343567
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5438503
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5436355
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5395017
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5378548
time: 0.80 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5397462
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5395017, upper bound: 27.5345851
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5388474
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5438503, upper bound: 27.5344886
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5418390
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5411491
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5391825
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5397372, upper bound: 27.5377179
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5397372
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5391825, upper bound: 27.5344487
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5411491, upper bound: 27.5387729
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5418390, upper bound: 27.5343567
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5438503
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5436355
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5395017
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5378548

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5176531, upper bound: 27.5167048
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5176531, upper bound: 27.5167048
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5168603, upper bound: 27.5167048
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169915
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168325
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168603
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168352
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.76 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5176531, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5176531, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5168603, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169915
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168325
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168603
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168352
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
Binary search (step 2): status=Status.VERIFIED, low=0.1093750, high=0.1250000, mid=0.1093750, abs_max=31.572500228881836
rel_dist={3: [-27.547796646245764, 27.54779664624577]}

## Binary search (step 3) starts
Candidate diff: 0.1171875


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940
time: 0.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5454382
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456764
time: 0.67 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456010
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5472606
time: 0.75 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.05 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.05
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5454382
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.05
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456764
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.05
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456010
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.05
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5472606

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5438520
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428002
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5441190
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428747, upper bound: 27.5427901
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428747
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5455850
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5438520, upper bound: 27.5428990
time: 0.74 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.15 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5438520
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428002
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5441190
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 3, lower bound: -27.5428747, upper bound: 27.5427901
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428747
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5455850
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 3, lower bound: -27.5438520, upper bound: 27.5428990

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5397462
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5395017, upper bound: 27.5345851
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5388474
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5438503, upper bound: 27.5344886
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5418390
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5411491
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5391825
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5397372, upper bound: 27.5377179
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5377179, upper bound: 27.5397372
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5344487
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5411491, upper bound: 27.5387729
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5397462, upper bound: 27.5343567
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5438503
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5436355
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5395017
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5397462, upper bound: 27.5378548
time: 0.76 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.09 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5397462
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5395017, upper bound: 27.5345851
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5388474
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5438503, upper bound: 27.5344886
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5418390
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5411491
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5391825
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5397372, upper bound: 27.5377179
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5377179, upper bound: 27.5397372
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5344487
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5411491, upper bound: 27.5387729
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5397462, upper bound: 27.5343567
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5438503
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5436355
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5395017
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5397462, upper bound: 27.5378548

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5169085, upper bound: 27.5167048
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169915
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168325
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168603
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168352
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.73 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5169085, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169915
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168325
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168603
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168352
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
Binary search (step 3): status=Status.VERIFIED, low=0.1171875, high=0.1250000, mid=0.1171875, abs_max=31.572500228881836
rel_dist={3: [-27.547796646245764, 27.54779664624577]}

## Binary search (step 4) starts
Candidate diff: 0.1210938


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940
time: 0.71 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.55 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5454382
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5456764
time: 0.74 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456010
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5472606
time: 0.76 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.13 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5454382
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5456764
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456010
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5472606

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5438520
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428990, upper bound: 27.5428002
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428747, upper bound: 27.5441190
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5427901
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428747
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5455850
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428990
time: 0.79 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.14 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5438520
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -27.5428990, upper bound: 27.5428002
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -27.5428747, upper bound: 27.5441190
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5427901
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428747
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5455850
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428990

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5397462
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5395017, upper bound: 27.5345851
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5436355, upper bound: 27.5388474
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5438503, upper bound: 27.5344886
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5418390
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5411491
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5391825
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5397372, upper bound: 27.5377179
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5377179, upper bound: 27.5397372
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5391825, upper bound: 27.5344487
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5411491, upper bound: 27.5387729
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5418390, upper bound: 27.5343567
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5438503
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5388474, upper bound: 27.5436355
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5395017
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5397462, upper bound: 27.5378548
time: 0.75 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5397462
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -27.5395017, upper bound: 27.5345851
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -27.5436355, upper bound: 27.5388474
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -27.5438503, upper bound: 27.5344886
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5418390
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5411491
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5391825
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -27.5397372, upper bound: 27.5377179
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -27.5377179, upper bound: 27.5397372
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -27.5391825, upper bound: 27.5344487
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -27.5411491, upper bound: 27.5387729
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -27.5418390, upper bound: 27.5343567
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5438503
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -27.5388474, upper bound: 27.5436355
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5395017
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -27.5397462, upper bound: 27.5378548

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5172478, upper bound: 27.5167048
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5176531, upper bound: 27.5167048
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5169915, upper bound: 27.5167048
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169915
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168325
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168603
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168352
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.72 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5172478, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5176531, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5169915, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169915
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168325
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168603
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168352
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.06
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
Binary search (step 4): status=Status.VERIFIED, low=0.1210938, high=0.1250000, mid=0.1210938, abs_max=31.572500228881836
rel_dist={3: [-27.547796646245764, 27.54779664624577]}

## Binary search (step 5) starts
Candidate diff: 0.1230469


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940
time: 0.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5454382
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456764
time: 0.65 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456010
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5472606
time: 0.80 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.14 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.14
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5454382
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.14
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456764
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.14
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456010
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.14
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5472606

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5438520
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428002
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5441190
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428747, upper bound: 27.5427901
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428747
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5455850
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5428990
time: 0.75 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.08 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5438520
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428002
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5441190
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 3, lower bound: -27.5428747, upper bound: 27.5427901
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428747
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5455850
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.08
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5428990

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5397462
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5345851
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5388474
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5395017, upper bound: 27.5344886
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5418390
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5411491
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5391825
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5397372, upper bound: 27.5377179
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5377179, upper bound: 27.5397372
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5391825, upper bound: 27.5344487
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5377179, upper bound: 27.5387729
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5391825, upper bound: 27.5343567
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5438503
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5436355
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5395017
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5378548
time: 0.61 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.87
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5397462
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.87
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5345851
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.87
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5388474
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.87
Output dim: 3, lower bound: -27.5395017, upper bound: 27.5344886
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.87
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5418390
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.87
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5411491
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.87
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5391825
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.87
Output dim: 3, lower bound: -27.5397372, upper bound: 27.5377179
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.87
Output dim: 3, lower bound: -27.5377179, upper bound: 27.5397372
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.87
Output dim: 3, lower bound: -27.5391825, upper bound: 27.5344487
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.87
Output dim: 3, lower bound: -27.5377179, upper bound: 27.5387729
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.87
Output dim: 3, lower bound: -27.5391825, upper bound: 27.5343567
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.87
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5438503
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.87
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5436355
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.87
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5395017
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.87
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5378548

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5176531, upper bound: 27.5167048
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5168603, upper bound: 27.5167048
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169915
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168325
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168603
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168352
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.78 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5176531, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5168603, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169915
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168325
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168603
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168352
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
Binary search (step 5): status=Status.VERIFIED, low=0.1230469, high=0.1250000, mid=0.1230469, abs_max=31.572500228881836
rel_dist={3: [-27.547796646245764, 27.54779664624577]}

## Binary search (step 6) starts
Candidate diff: 0.1240234


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940
time: 0.78 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.62 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5454382
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456764
time: 0.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456010
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5472606
time: 0.76 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.09
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5454382
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.09
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456764
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.09
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456010
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.09
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5472606

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5438520
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428002
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5441190
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428747, upper bound: 27.5427901
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428747
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5455850
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5438520, upper bound: 27.5428990
time: 0.77 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.09 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5438520
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428002
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5441190
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 3, lower bound: -27.5428747, upper bound: 27.5427901
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428747
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5455850
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.09
Output dim: 3, lower bound: -27.5438520, upper bound: 27.5428990

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5397462
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5345851
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5388474
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5438503, upper bound: 27.5344886
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5418390
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5411491
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5391825
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5377179
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5377179, upper bound: 27.5397372
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5391825, upper bound: 27.5344487
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5387729
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5418390, upper bound: 27.5343567
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5438503
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5436355
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5395017
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5378548
time: 0.63 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5397462
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5345851
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5388474
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 3, lower bound: -27.5438503, upper bound: 27.5344886
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5418390
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5411491
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5391825
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5377179
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 3, lower bound: -27.5377179, upper bound: 27.5397372
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 3, lower bound: -27.5391825, upper bound: 27.5344487
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5387729
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 3, lower bound: -27.5418390, upper bound: 27.5343567
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5438503
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5436355
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5395017
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.82
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5378548

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5169085, upper bound: 27.5167048
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169915
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168325
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168603
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168352
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.73 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5169085, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169915
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168325
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168603
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168352
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.24
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
Binary search (step 6): status=Status.VERIFIED, low=0.1240234, high=0.1250000, mid=0.1240234, abs_max=31.572500228881836
rel_dist={3: [-27.547796646245764, 27.54779664624577]}

## Binary search (step 7) starts
Candidate diff: 0.1245117


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940
time: 0.77 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5454382
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456764
time: 0.66 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456010
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5472606
time: 0.80 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.13 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5454382
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456764
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456010
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5472606

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428990, upper bound: 27.5438520
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5455850, upper bound: 27.5428002
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5441190
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5427901
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428747
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5455850
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5438520, upper bound: 27.5428990
time: 0.79 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.13 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 3, lower bound: -27.5428990, upper bound: 27.5438520
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 3, lower bound: -27.5455850, upper bound: 27.5428002
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5441190
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5427901
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428747
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5455850
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.13
Output dim: 3, lower bound: -27.5438520, upper bound: 27.5428990

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5397462
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5345851
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5388474
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5438503, upper bound: 27.5344886
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5418390
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5411491
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5391825
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5397372, upper bound: 27.5377179
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5377179, upper bound: 27.5397372
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5388474, upper bound: 27.5344487
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5411491, upper bound: 27.5387729
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5418390, upper bound: 27.5343567
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5438503
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5436355
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5395017
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5397462, upper bound: 27.5378548
time: 0.79 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.09 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5397462
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5345851
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5388474
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5438503, upper bound: 27.5344886
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5418390
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5411491
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5391825
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5397372, upper bound: 27.5377179
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5377179, upper bound: 27.5397372
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5388474, upper bound: 27.5344487
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5411491, upper bound: 27.5387729
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5418390, upper bound: 27.5343567
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5438503
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5436355
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5395017
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -27.5397462, upper bound: 27.5378548

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5169085, upper bound: 27.5167048
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169915
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168325
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168603
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168352
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.72 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5169085, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169915
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168325
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168603
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168352
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.26
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
Binary search (step 7): status=Status.VERIFIED, low=0.1245117, high=0.1250000, mid=0.1245117, abs_max=31.572500228881836
rel_dist={3: [-27.547796646245764, 27.547796646245764]}

## Binary search (step 8) starts
Candidate diff: 0.1247559


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940
time: 0.76 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.73
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.73
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5454382
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456764
time: 0.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456010
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5472606
time: 0.75 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.06 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5454382
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456764
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456010
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5472606

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5438520
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5455850, upper bound: 27.5428002
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5441190
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5440239, upper bound: 27.5427901
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5441190, upper bound: 27.5428747
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5455850
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5438520, upper bound: 27.5428990
time: 0.78 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.93 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5438520
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 3, lower bound: -27.5455850, upper bound: 27.5428002
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5441190
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 3, lower bound: -27.5440239, upper bound: 27.5427901
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 3, lower bound: -27.5441190, upper bound: 27.5428747
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5455850
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 3, lower bound: -27.5438520, upper bound: 27.5428990

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5397462
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5345851
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5388474
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5438503, upper bound: 27.5344886
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5418390
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5411491
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5391825
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5377179
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5377179, upper bound: 27.5397372
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5391825, upper bound: 27.5344487
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5411491, upper bound: 27.5387729
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5418390, upper bound: 27.5343567
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5438503
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5436355
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344886, upper bound: 27.5395017
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5397462, upper bound: 27.5378548
time: 0.83 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5397462
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5345851
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5388474
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -27.5438503, upper bound: 27.5344886
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5418390
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5411491
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5391825
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5377179
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -27.5377179, upper bound: 27.5397372
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -27.5391825, upper bound: 27.5344487
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -27.5411491, upper bound: 27.5387729
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -27.5418390, upper bound: 27.5343567
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5438503
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5436355
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -27.5344886, upper bound: 27.5395017
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -27.5397462, upper bound: 27.5378548

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169915
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168325
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168603
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168352
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
time: 0.66 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169915
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168325
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168603
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5168352
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5176531
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5172478
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5169085
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.90
Output dim: 3, lower bound: -27.5167048, upper bound: 27.5167048
Binary search (step 8): status=Status.VERIFIED, low=0.1247559, high=0.1250000, mid=0.1247559, abs_max=31.572500228881836
rel_dist={3: [-27.547796646245764, 27.54779664624577]}

## Binary search (step 9) starts
Candidate diff: 0.1248779


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940
time: 0.72 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5460616
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 3, lower bound: -27.5460616, upper bound: 27.5476940

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5454382
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5456764
time: 0.73 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456010
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5472606
time: 0.76 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.03 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5454382
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 3, lower bound: -27.5456010, upper bound: 27.5456764
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5456010
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 3, lower bound: -27.5454382, upper bound: 27.5472606

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428990, upper bound: 27.5438520
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428002
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5441190
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5427901
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428747
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5455850
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5438520, upper bound: 27.5428990
time: 0.79 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.97
Output dim: 3, lower bound: -27.5428990, upper bound: 27.5438520
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.97
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428002
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.97
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5441190
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.97
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5427901
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.97
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5440239
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.97
Output dim: 3, lower bound: -27.5427901, upper bound: 27.5428747
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.97
Output dim: 3, lower bound: -27.5428002, upper bound: 27.5455850
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.97
Output dim: 3, lower bound: -27.5438520, upper bound: 27.5428990

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5397462
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5345851
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5388474
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5397372, upper bound: 27.5344886
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5418390
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5411491
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5391825
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5377179
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344886, upper bound: 27.5397372
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344886, upper bound: 27.5344487
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5411491, upper bound: 27.5387729
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5418390, upper bound: 27.5343567
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5438503
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5436355
time: 0.85 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5397462
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5345851
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -27.5378548, upper bound: 27.5388474
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -27.5397372, upper bound: 27.5344886
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5418390
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5411491
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -27.5344487, upper bound: 27.5391825
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5377179
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -27.5344886, upper bound: 27.5397372
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -27.5344886, upper bound: 27.5344487
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -27.5411491, upper bound: 27.5387729
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -27.5418390, upper bound: 27.5343567
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -27.5343567, upper bound: 27.5438503
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -27.5387729, upper bound: 27.5436355
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 3, lower bound: -27.5438520, upper bound: 27.5428990
Binary search (step 9): status=Status.UNKNOWN, low=0.1247559, high=0.1248779, mid=0.1248779, abs_max=31.572500228881836
rel_dist={3: [-27.547796646245764, 27.54779664624577]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.124755859375
execution time: 1125.09 seconds
