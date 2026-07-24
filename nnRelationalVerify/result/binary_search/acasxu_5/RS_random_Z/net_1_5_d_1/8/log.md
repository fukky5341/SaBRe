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
execution time: IAR + LP analysis = 2.48 + 1.96 = 4.45 seconds
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
Binary search time: 72.05 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1123.50 seconds

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5437182, upper bound: 27.5461158
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5437182, upper bound: 27.5437182
time: 0.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.48 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.48
Output dim: 3, lower bound: -27.5437182, upper bound: 27.5461158
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.48
Output dim: 3, lower bound: -27.5437182, upper bound: 27.5437182

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5414413, upper bound: 27.5437029
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5414413, upper bound: 27.5442124
time: 0.72 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5412470, upper bound: 27.5413384
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5413384, upper bound: 27.5412470
time: 0.83 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.89 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.89
Output dim: 3, lower bound: -27.5414413, upper bound: 27.5437029
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.89
Output dim: 3, lower bound: -27.5414413, upper bound: 27.5442124
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.89
Output dim: 3, lower bound: -27.5412470, upper bound: 27.5413384
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.89
Output dim: 3, lower bound: -27.5413384, upper bound: 27.5412470

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5389254, upper bound: 27.5415635
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5390096, upper bound: 27.5417187
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5367572, upper bound: 27.5424683
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5367572, upper bound: 27.5424101
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5348713, upper bound: 27.5387375
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343426, upper bound: 27.5344454
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343426, upper bound: 27.5386262
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5347845, upper bound: 27.5343426
time: 0.70 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.87 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.87
Output dim: 3, lower bound: -27.5389254, upper bound: 27.5415635
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.87
Output dim: 3, lower bound: -27.5390096, upper bound: 27.5417187
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.87
Output dim: 3, lower bound: -27.5367572, upper bound: 27.5424683
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.87
Output dim: 3, lower bound: -27.5367572, upper bound: 27.5424101
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.87
Output dim: 3, lower bound: -27.5348713, upper bound: 27.5387375
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.87
Output dim: 3, lower bound: -27.5343426, upper bound: 27.5344454
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.87
Output dim: 3, lower bound: -27.5343426, upper bound: 27.5386262
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.87
Output dim: 3, lower bound: -27.5347845, upper bound: 27.5343426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5384283, upper bound: 27.5399440
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5385025, upper bound: 27.5410558
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5380407, upper bound: 27.5393756
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5381295, upper bound: 27.5411592
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5331064, upper bound: 27.5389266
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5331064, upper bound: 27.5332297
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5074007, upper bound: 27.5079981
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5074007, upper bound: 27.5080111
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5323405, upper bound: 27.5367343
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5323405, upper bound: 27.5367867
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5337547, upper bound: 27.5339607
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5337547, upper bound: 27.5339131
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5337998, upper bound: 27.5380303
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5337998, upper bound: 27.5380828
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5338778, upper bound: 27.5338778
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5342507, upper bound: 27.5338778
time: 0.70 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 3, lower bound: -27.5384283, upper bound: 27.5399440
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 3, lower bound: -27.5385025, upper bound: 27.5410558
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 3, lower bound: -27.5380407, upper bound: 27.5393756
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 3, lower bound: -27.5381295, upper bound: 27.5411592
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 3, lower bound: -27.5331064, upper bound: 27.5389266
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 3, lower bound: -27.5331064, upper bound: 27.5332297
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.90
Output dim: 3, lower bound: -27.5074007, upper bound: 27.5079981
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.90
Output dim: 3, lower bound: -27.5074007, upper bound: 27.5080111
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 3, lower bound: -27.5323405, upper bound: 27.5367343
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 3, lower bound: -27.5323405, upper bound: 27.5367867
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 3, lower bound: -27.5337547, upper bound: 27.5339607
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 3, lower bound: -27.5337547, upper bound: 27.5339131
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 3, lower bound: -27.5337998, upper bound: 27.5380303
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 3, lower bound: -27.5337998, upper bound: 27.5380828
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 3, lower bound: -27.5338778, upper bound: 27.5338778
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 3, lower bound: -27.5342507, upper bound: 27.5338778

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5303715, upper bound: 27.5365286
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5303715, upper bound: 27.5304157
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5149535, upper bound: 27.5149535
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5149535, upper bound: 27.5149535
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5379211, upper bound: 27.5392344
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5316680, upper bound: 27.5392344
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5303715, upper bound: 27.5391617
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5303715, upper bound: 27.5387750
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5328213, upper bound: 27.5368896
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5328213, upper bound: 27.5388179
time: 0.75 seconds

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
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5161262, upper bound: 27.5164064
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5161262, upper bound: 27.5164064
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5319925, upper bound: 27.5364243
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5319925, upper bound: 27.5364473
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5318589, upper bound: 27.5364862
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5318589, upper bound: 27.5322299
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4893871, upper bound: 27.4894678
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4893871, upper bound: 27.4894678
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5155368, upper bound: 27.5157409
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5155368, upper bound: 27.5157409
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4894763, upper bound: 27.4893871
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4893871, upper bound: 27.4893871
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5331924, upper bound: 27.5374818
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5331924, upper bound: 27.5375678
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5332562, upper bound: 27.5332562
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5332562, upper bound: 27.5332562
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5333419, upper bound: 27.5333419
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5333419, upper bound: 27.5333419
time: 0.71 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5303715, upper bound: 27.5365286
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5303715, upper bound: 27.5304157
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5149535, upper bound: 27.5149535
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5149535, upper bound: 27.5149535
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5379211, upper bound: 27.5392344
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5316680, upper bound: 27.5392344
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5303715, upper bound: 27.5391617
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5303715, upper bound: 27.5387750
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5328213, upper bound: 27.5368896
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5328213, upper bound: 27.5388179
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5161262, upper bound: 27.5164064
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5161262, upper bound: 27.5164064
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5319925, upper bound: 27.5364243
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5319925, upper bound: 27.5364473
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5318589, upper bound: 27.5364862
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5318589, upper bound: 27.5322299
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.4893871, upper bound: 27.4894678
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.4893871, upper bound: 27.4894678
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5155368, upper bound: 27.5157409
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5155368, upper bound: 27.5157409
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.4894763, upper bound: 27.4893871
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.4893871, upper bound: 27.4893871
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5331924, upper bound: 27.5374818
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5331924, upper bound: 27.5375678
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5332562, upper bound: 27.5332562
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5332562, upper bound: 27.5332562
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5333419, upper bound: 27.5333419
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 3, lower bound: -27.5333419, upper bound: 27.5333419

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5300845, upper bound: 27.5365222
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5300845, upper bound: 27.5365286
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5055122, upper bound: 27.5055122
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5055122, upper bound: 27.5055122
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5077092, upper bound: 27.5066663
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5066663, upper bound: 27.5066663
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5311838, upper bound: 27.5386631
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5311838, upper bound: 27.5311838
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5299359, upper bound: 27.5387562
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5299359, upper bound: 27.5336513
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5055122, upper bound: 27.5055122
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5055122, upper bound: 27.5055122
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5324200, upper bound: 27.5366656
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5324200, upper bound: 27.5338240
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5304953, upper bound: 27.5365976
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5304953, upper bound: 27.5366000
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5313054, upper bound: 27.5359793
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5313054, upper bound: 27.5359700
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5149239, upper bound: 27.5149239
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5149239, upper bound: 27.5149239
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145776, upper bound: 27.5146446
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145776, upper bound: 27.5146446
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5155368, upper bound: 27.5155368
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5155368, upper bound: 27.5155368
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5326790, upper bound: 27.5372107
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5326790, upper bound: 27.5342117
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5326790, upper bound: 27.5326790
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5326790, upper bound: 27.5326790
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4893743, upper bound: 27.4893743
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4894641, upper bound: 27.4893743
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5326790, upper bound: 27.5326790
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5326790, upper bound: 27.5326790
time: 0.83 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5300845, upper bound: 27.5365222
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5300845, upper bound: 27.5365286
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5055122, upper bound: 27.5055122
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5055122, upper bound: 27.5055122
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5077092, upper bound: 27.5066663
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5066663, upper bound: 27.5066663
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5311838, upper bound: 27.5386631
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5311838, upper bound: 27.5311838
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5299359, upper bound: 27.5387562
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5299359, upper bound: 27.5336513
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5055122, upper bound: 27.5055122
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5055122, upper bound: 27.5055122
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5324200, upper bound: 27.5366656
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5324200, upper bound: 27.5338240
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5304953, upper bound: 27.5365976
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5304953, upper bound: 27.5366000
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5313054, upper bound: 27.5359793
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5313054, upper bound: 27.5359700
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5149239, upper bound: 27.5149239
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5149239, upper bound: 27.5149239
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5145776, upper bound: 27.5146446
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5145776, upper bound: 27.5146446
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5155368, upper bound: 27.5155368
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5155368, upper bound: 27.5155368
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5326790, upper bound: 27.5372107
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5326790, upper bound: 27.5342117
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5326790, upper bound: 27.5326790
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5326790, upper bound: 27.5326790
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.4893743, upper bound: 27.4893743
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.4894641, upper bound: 27.4893743
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5326790, upper bound: 27.5326790
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 3, lower bound: -27.5326790, upper bound: 27.5326790

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5055122, upper bound: 27.5055122
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5055122, upper bound: 27.5055122
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5346178
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5298837
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5355766
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5146073, upper bound: 27.5146073
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5146073, upper bound: 27.5146073
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5054735, upper bound: 27.5054735
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5054735, upper bound: 27.5054735
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5297302, upper bound: 27.5336375
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5297302, upper bound: 27.5297302
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4869266, upper bound: 27.4869266
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4869266, upper bound: 27.4869266
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4869266, upper bound: 27.4869266
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4869266, upper bound: 27.4869266
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5152349
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5152349
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5342597
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5359809
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5149239, upper bound: 27.5149239
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5149239, upper bound: 27.5149239
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5308528, upper bound: 27.5355990
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5308528, upper bound: 27.5321884
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4893743, upper bound: 27.4893743
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4893743, upper bound: 27.4893743
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4893743, upper bound: 27.4893743
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4893743, upper bound: 27.4893743
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5308528, upper bound: 27.5308528
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5308528, upper bound: 27.5308528
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
time: 0.73 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5055122, upper bound: 27.5055122
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5055122, upper bound: 27.5055122
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5346178
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5298837
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5355766
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5146073, upper bound: 27.5146073
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5146073, upper bound: 27.5146073
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5054735, upper bound: 27.5054735
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5054735, upper bound: 27.5054735
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5297302, upper bound: 27.5336375
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5297302, upper bound: 27.5297302
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.4869266, upper bound: 27.4869266
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.4869266, upper bound: 27.4869266
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.4869266, upper bound: 27.4869266
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.4869266, upper bound: 27.4869266
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5152349
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5152349
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5342597
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5359809
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5149239, upper bound: 27.5149239
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5149239, upper bound: 27.5149239
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5308528, upper bound: 27.5355990
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5308528, upper bound: 27.5321884
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.4893743, upper bound: 27.4893743
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.4893743, upper bound: 27.4893743
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.4893743, upper bound: 27.4893743
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.4893743, upper bound: 27.4893743
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5308528, upper bound: 27.5308528
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5308528, upper bound: 27.5308528
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5343591
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5146016, upper bound: 27.5146016
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5146016, upper bound: 27.5146016
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5146016, upper bound: 27.5146016
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5146016, upper bound: 27.5146016
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5148912
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5148912
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145776, upper bound: 27.5145776
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145776, upper bound: 27.5145776
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5309970
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5303238
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.82 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5343591
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.5146016, upper bound: 27.5146016
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.5146016, upper bound: 27.5146016
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.5146016, upper bound: 27.5146016
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.5146016, upper bound: 27.5146016
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5148912
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5148912
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.5145776, upper bound: 27.5145776
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.5145776, upper bound: 27.5145776
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5309970
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5303238
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.00
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.75 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 4.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.05
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.05
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.05
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.05
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.05
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.05
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.05
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.05
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.05
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.05
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.05
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.05
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.05
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.05
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.05
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.05
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.05
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.05
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.05
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.05
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
Binary search (step 0): status=Status.VERIFIED, low=0.0625000, high=0.1250000, mid=0.0625000, abs_max=31.572500228881836
rel_dist={3: [-27.547796646245764, 27.54779664624577]}

## Binary search (step 1) starts
Candidate diff: 0.0937500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5433295, upper bound: 27.5474341
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5433295, upper bound: 27.5433295
time: 0.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.33 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.33
Output dim: 3, lower bound: -27.5433295, upper bound: 27.5474341
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.33
Output dim: 3, lower bound: -27.5433295, upper bound: 27.5433295

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5195159, upper bound: 27.5205087
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5195159, upper bound: 27.5205087
time: 0.75 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5424195, upper bound: 27.5424195
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5424195, upper bound: 27.5427551
time: 0.81 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.88 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.88
Output dim: 3, lower bound: -27.5195159, upper bound: 27.5205087
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.88
Output dim: 3, lower bound: -27.5195159, upper bound: 27.5205087
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.88
Output dim: 3, lower bound: -27.5424195, upper bound: 27.5424195
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.88
Output dim: 3, lower bound: -27.5424195, upper bound: 27.5427551

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5194858, upper bound: 27.5204825
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5194858, upper bound: 27.5203521
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5195159, upper bound: 27.5199428
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5195159, upper bound: 27.5205087
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5398200, upper bound: 27.5398200
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5398200, upper bound: 27.5401823
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5360994, upper bound: 27.5410853
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5360994, upper bound: 27.5407579
time: 0.77 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.85 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.85
Output dim: 3, lower bound: -27.5194858, upper bound: 27.5204825
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.85
Output dim: 3, lower bound: -27.5194858, upper bound: 27.5203521
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.85
Output dim: 3, lower bound: -27.5195159, upper bound: 27.5199428
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.85
Output dim: 3, lower bound: -27.5195159, upper bound: 27.5205087
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.85
Output dim: 3, lower bound: -27.5398200, upper bound: 27.5398200
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.85
Output dim: 3, lower bound: -27.5398200, upper bound: 27.5401823
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.85
Output dim: 3, lower bound: -27.5360994, upper bound: 27.5410853
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.85
Output dim: 3, lower bound: -27.5360994, upper bound: 27.5407579

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5194858, upper bound: 27.5194858
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5194858, upper bound: 27.5204825
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5194858, upper bound: 27.5198997
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5194858, upper bound: 27.5203521
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5190712, upper bound: 27.5190712
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5190712, upper bound: 27.5197136
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5355888, upper bound: 27.5365933
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5355888, upper bound: 27.5390881
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5127610, upper bound: 27.5127610
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5127610, upper bound: 27.5127610
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5134498, upper bound: 27.5134498
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5134498, upper bound: 27.5134498
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5344627, upper bound: 27.5380739
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5421382, upper bound: 27.5384684
time: 0.77 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.89
Output dim: 3, lower bound: -27.5194858, upper bound: 27.5194858
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -27.5194858, upper bound: 27.5204825
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.89
Output dim: 3, lower bound: -27.5194858, upper bound: 27.5198997
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -27.5194858, upper bound: 27.5203521
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.89
Output dim: 3, lower bound: -27.5190712, upper bound: 27.5190712
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.89
Output dim: 3, lower bound: -27.5190712, upper bound: 27.5197136
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -27.5355888, upper bound: 27.5365933
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -27.5355888, upper bound: 27.5390881
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.89
Output dim: 3, lower bound: -27.5127610, upper bound: 27.5127610
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.89
Output dim: 3, lower bound: -27.5127610, upper bound: 27.5127610
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.89
Output dim: 3, lower bound: -27.5134498, upper bound: 27.5134498
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.89
Output dim: 3, lower bound: -27.5134498, upper bound: 27.5134498
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -27.5344627, upper bound: 27.5380739
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 3, lower bound: -27.5421382, upper bound: 27.5384684

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5190400, upper bound: 27.5190400
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5190400, upper bound: 27.5196856
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5194858, upper bound: 27.5194858
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5194858, upper bound: 27.5203521
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5340551, upper bound: 27.5349224
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5340551, upper bound: 27.5340551
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5350720, upper bound: 27.5382366
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5350720, upper bound: 27.5350720
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5339406, upper bound: 27.5378340
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5339406, upper bound: 27.5349200
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5339406, upper bound: 27.5381398
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5339406, upper bound: 27.5349983
time: 0.87 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.92
Output dim: 3, lower bound: -27.5190400, upper bound: 27.5190400
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.92
Output dim: 3, lower bound: -27.5190400, upper bound: 27.5196856
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.92
Output dim: 3, lower bound: -27.5194858, upper bound: 27.5194858
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.92
Output dim: 3, lower bound: -27.5194858, upper bound: 27.5203521
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.92
Output dim: 3, lower bound: -27.5340551, upper bound: 27.5349224
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.92
Output dim: 3, lower bound: -27.5340551, upper bound: 27.5340551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.92
Output dim: 3, lower bound: -27.5350720, upper bound: 27.5382366
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.92
Output dim: 3, lower bound: -27.5350720, upper bound: 27.5350720
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.92
Output dim: 3, lower bound: -27.5339406, upper bound: 27.5378340
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.92
Output dim: 3, lower bound: -27.5339406, upper bound: 27.5349200
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.92
Output dim: 3, lower bound: -27.5339406, upper bound: 27.5381398
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.92
Output dim: 3, lower bound: -27.5339406, upper bound: 27.5349983

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5190400, upper bound: 27.5190400
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5190400, upper bound: 27.5195737
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5180786, upper bound: 27.5180473
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5180786, upper bound: 27.5180473
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5180473, upper bound: 27.5180473
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5180473, upper bound: 27.5180473
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5184625, upper bound: 27.5184625
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5184625, upper bound: 27.5184625
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5326790, upper bound: 27.5326790
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5326790, upper bound: 27.5326790
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5315152, upper bound: 27.5358130
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5315152, upper bound: 27.5319041
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5327112, upper bound: 27.5327134
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5327112, upper bound: 27.5331938
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5315152, upper bound: 27.5362230
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5315152, upper bound: 27.5324534
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5315152, upper bound: 27.5322828
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5315152, upper bound: 27.5315152
time: 0.72 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.92 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 3, lower bound: -27.5190400, upper bound: 27.5190400
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 3, lower bound: -27.5190400, upper bound: 27.5195737
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 3, lower bound: -27.5180786, upper bound: 27.5180473
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 3, lower bound: -27.5180786, upper bound: 27.5180473
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 3, lower bound: -27.5180473, upper bound: 27.5180473
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 3, lower bound: -27.5180473, upper bound: 27.5180473
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 3, lower bound: -27.5184625, upper bound: 27.5184625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 3, lower bound: -27.5184625, upper bound: 27.5184625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -27.5326790, upper bound: 27.5326790
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -27.5326790, upper bound: 27.5326790
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -27.5315152, upper bound: 27.5358130
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -27.5315152, upper bound: 27.5319041
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -27.5327112, upper bound: 27.5327134
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -27.5327112, upper bound: 27.5331938
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -27.5315152, upper bound: 27.5362230
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -27.5315152, upper bound: 27.5324534
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -27.5315152, upper bound: 27.5322828
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 3, lower bound: -27.5315152, upper bound: 27.5315152

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4988936, upper bound: 27.4988936
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4988936, upper bound: 27.4988936
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5300905, upper bound: 27.5305615
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5300905, upper bound: 27.5300905
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5308528, upper bound: 27.5308528
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5308528, upper bound: 27.5317637
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145776, upper bound: 27.5145776
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145776, upper bound: 27.5145776
time: 0.79 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.97 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.4988936, upper bound: 27.4988936
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.4988936, upper bound: 27.4988936
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5300905, upper bound: 27.5305615
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5300905, upper bound: 27.5300905
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5308528, upper bound: 27.5308528
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5308528, upper bound: 27.5317637
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.4858577, upper bound: 27.4858577
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5145776, upper bound: 27.5145776
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 3, lower bound: -27.5145776, upper bound: 27.5145776

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5299052
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145776, upper bound: 27.5145776
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145776, upper bound: 27.5145776
time: 0.71 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.03 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.03
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.03
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5299052
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.03
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.03
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.03
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.03
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.03
Output dim: 3, lower bound: -27.5145776, upper bound: 27.5145776
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.03
Output dim: 3, lower bound: -27.5145776, upper bound: 27.5145776

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.61 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.72 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.72
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.72
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.72
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.72
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.72
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.72
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.72
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.72
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
Binary search (step 1): status=Status.VERIFIED, low=0.0937500, high=0.1250000, mid=0.0937500, abs_max=31.572500228881836
rel_dist={3: [-27.547796646245764, 27.54779664624577]}

## Binary search (step 2) starts
Candidate diff: 0.1093750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5462906, upper bound: 27.5462906
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5462906, upper bound: 27.5462906
time: 0.59 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.20 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.20
Output dim: 3, lower bound: -27.5462906, upper bound: 27.5462906
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.20
Output dim: 3, lower bound: -27.5462906, upper bound: 27.5462906

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
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5375419, upper bound: 27.5424676
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5375419, upper bound: 27.5459224
time: 0.76 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5459224, upper bound: 27.5375419
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5424676, upper bound: 27.5459007
time: 0.72 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.76 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.76
Output dim: 3, lower bound: -27.5375419, upper bound: 27.5424676
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.76
Output dim: 3, lower bound: -27.5375419, upper bound: 27.5459224
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.76
Output dim: 3, lower bound: -27.5459224, upper bound: 27.5375419
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.76
Output dim: 3, lower bound: -27.5424676, upper bound: 27.5459007

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5370223, upper bound: 27.5406163
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5370223, upper bound: 27.5366243
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5351815, upper bound: 27.5438612
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5351652, upper bound: 27.5395912
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5411577, upper bound: 27.5358406
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5411577, upper bound: 27.5356201
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5422789, upper bound: 27.5434579
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5422789, upper bound: 27.5458867
time: 0.66 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.73 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.73
Output dim: 3, lower bound: -27.5370223, upper bound: 27.5406163
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.73
Output dim: 3, lower bound: -27.5370223, upper bound: 27.5366243
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.73
Output dim: 3, lower bound: -27.5351815, upper bound: 27.5438612
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.73
Output dim: 3, lower bound: -27.5351652, upper bound: 27.5395912
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.73
Output dim: 3, lower bound: -27.5411577, upper bound: 27.5358406
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.73
Output dim: 3, lower bound: -27.5411577, upper bound: 27.5356201
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.73
Output dim: 3, lower bound: -27.5422789, upper bound: 27.5434579
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.73
Output dim: 3, lower bound: -27.5422789, upper bound: 27.5458867

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5348368, upper bound: 27.5388829
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5348369, upper bound: 27.5387403
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5349249, upper bound: 27.5344296
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5402268, upper bound: 27.5344005
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5327084, upper bound: 27.5418095
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5324262, upper bound: 27.5418427
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5088501, upper bound: 27.5096184
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5088501, upper bound: 27.5101215
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5411564, upper bound: 27.5352913
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5365351, upper bound: 27.5351118
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5194419, upper bound: 27.5194419
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5194419, upper bound: 27.5194419
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5403634, upper bound: 27.5409507
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5407437, upper bound: 27.5412589
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5403634, upper bound: 27.5409507
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5407468, upper bound: 27.5437598
time: 0.83 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.12
Output dim: 3, lower bound: -27.5348368, upper bound: 27.5388829
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.12
Output dim: 3, lower bound: -27.5348369, upper bound: 27.5387403
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.12
Output dim: 3, lower bound: -27.5349249, upper bound: 27.5344296
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.12
Output dim: 3, lower bound: -27.5402268, upper bound: 27.5344005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.12
Output dim: 3, lower bound: -27.5327084, upper bound: 27.5418095
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.12
Output dim: 3, lower bound: -27.5324262, upper bound: 27.5418427
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.12
Output dim: 3, lower bound: -27.5088501, upper bound: 27.5096184
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.12
Output dim: 3, lower bound: -27.5088501, upper bound: 27.5101215
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.12
Output dim: 3, lower bound: -27.5411564, upper bound: 27.5352913
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.12
Output dim: 3, lower bound: -27.5365351, upper bound: 27.5351118
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.12
Output dim: 3, lower bound: -27.5194419, upper bound: 27.5194419
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.12
Output dim: 3, lower bound: -27.5194419, upper bound: 27.5194419
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.12
Output dim: 3, lower bound: -27.5403634, upper bound: 27.5409507
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.12
Output dim: 3, lower bound: -27.5407437, upper bound: 27.5412589
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.12
Output dim: 3, lower bound: -27.5403634, upper bound: 27.5409507
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.12
Output dim: 3, lower bound: -27.5407468, upper bound: 27.5437598

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5334741, upper bound: 27.5372923
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5334113, upper bound: 27.5372643
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5322747, upper bound: 27.5364873
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5321296, upper bound: 27.5361908
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5373115, upper bound: 27.5340848
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5340848, upper bound: 27.5341832
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5331104, upper bound: 27.5326646
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5379831, upper bound: 27.5326670
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5069222, upper bound: 27.5084535
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5069222, upper bound: 27.5089609
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5303715, upper bound: 27.5391617
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5303715, upper bound: 27.5395672
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5391560, upper bound: 27.5334948
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5420373, upper bound: 27.5335085
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5130911, upper bound: 27.5130911
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5130911, upper bound: 27.5130911
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5320817, upper bound: 27.5385056
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5320817, upper bound: 27.5330444
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5193962, upper bound: 27.5188633
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5194372, upper bound: 27.5188633
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343089, upper bound: 27.5400250
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343089, upper bound: 27.5349944
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5394617, upper bound: 27.5411372
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5394617, upper bound: 27.5419835
time: 0.85 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5334741, upper bound: 27.5372923
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5334113, upper bound: 27.5372643
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5322747, upper bound: 27.5364873
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5321296, upper bound: 27.5361908
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5373115, upper bound: 27.5340848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5340848, upper bound: 27.5341832
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5331104, upper bound: 27.5326646
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5379831, upper bound: 27.5326670
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5069222, upper bound: 27.5084535
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5069222, upper bound: 27.5089609
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5303715, upper bound: 27.5391617
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5303715, upper bound: 27.5395672
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5391560, upper bound: 27.5334948
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5420373, upper bound: 27.5335085
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5130911, upper bound: 27.5130911
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5130911, upper bound: 27.5130911
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5320817, upper bound: 27.5385056
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5320817, upper bound: 27.5330444
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5193962, upper bound: 27.5188633
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5194372, upper bound: 27.5188633
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5343089, upper bound: 27.5400250
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5343089, upper bound: 27.5349944
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5394617, upper bound: 27.5411372
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 3, lower bound: -27.5394617, upper bound: 27.5419835

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5329560, upper bound: 27.5370019
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5329098, upper bound: 27.5370040
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5329560, upper bound: 27.5369763
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5329098, upper bound: 27.5370073
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5155368, upper bound: 27.5155368
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5155368, upper bound: 27.5155368
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5317143, upper bound: 27.5359527
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5317143, upper bound: 27.5344230
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5356110, upper bound: 27.5335597
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5335566, upper bound: 27.5335566
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5317264, upper bound: 27.5322856
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5317264, upper bound: 27.5317264
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5161262, upper bound: 27.5161262
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5161262, upper bound: 27.5161262
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5161262, upper bound: 27.5161831
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5161262, upper bound: 27.5161757
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5149186, upper bound: 27.5149186
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5149186, upper bound: 27.5149186
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5300845, upper bound: 27.5316313
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5300845, upper bound: 27.5395672
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5114717, upper bound: 27.5114717
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5114717, upper bound: 27.5114717
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5300845, upper bound: 27.5302508
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5315184, upper bound: 27.5301226
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5069222, upper bound: 27.5069222
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5069222, upper bound: 27.5069222
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5301098, upper bound: 27.5310742
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5300845, upper bound: 27.5300845
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5338848, upper bound: 27.5397105
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5338848, upper bound: 27.5378335
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5329098, upper bound: 27.5329246
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5329098, upper bound: 27.5332963
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5114717, upper bound: 27.5114717
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5114717, upper bound: 27.5114717
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5329098, upper bound: 27.5382638
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5370019, upper bound: 27.5333326
time: 0.74 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5329560, upper bound: 27.5370019
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5329098, upper bound: 27.5370040
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5329560, upper bound: 27.5369763
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5329098, upper bound: 27.5370073
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5155368, upper bound: 27.5155368
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5155368, upper bound: 27.5155368
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5317143, upper bound: 27.5359527
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5317143, upper bound: 27.5344230
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5356110, upper bound: 27.5335597
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5335566, upper bound: 27.5335566
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5317264, upper bound: 27.5322856
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5317264, upper bound: 27.5317264
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5161262, upper bound: 27.5161262
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5161262, upper bound: 27.5161262
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5161262, upper bound: 27.5161831
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5161262, upper bound: 27.5161757
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5149186, upper bound: 27.5149186
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5149186, upper bound: 27.5149186
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5300845, upper bound: 27.5316313
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5300845, upper bound: 27.5395672
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5114717, upper bound: 27.5114717
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5114717, upper bound: 27.5114717
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5300845, upper bound: 27.5302508
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5315184, upper bound: 27.5301226
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5069222, upper bound: 27.5069222
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5069222, upper bound: 27.5069222
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5301098, upper bound: 27.5310742
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5300845, upper bound: 27.5300845
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5338848, upper bound: 27.5397105
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5338848, upper bound: 27.5378335
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5329098, upper bound: 27.5329246
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5329098, upper bound: 27.5332963
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5114717, upper bound: 27.5114717
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5114717, upper bound: 27.5114717
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5329098, upper bound: 27.5382638
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -27.5370019, upper bound: 27.5333326

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4989210, upper bound: 27.4989210
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4989210, upper bound: 27.4989210
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5346178
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5341377
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5183871, upper bound: 27.5183871
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5183871, upper bound: 27.5183871
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4989210, upper bound: 27.5017485
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4989210, upper bound: 27.5017485
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4889254, upper bound: 27.4889254
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4889254, upper bound: 27.4889254
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5297302, upper bound: 27.5317432
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5297302, upper bound: 27.5297302
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4902814, upper bound: 27.4902814
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4902814, upper bound: 27.4902814
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4902814, upper bound: 27.4902814
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4902814, upper bound: 27.4902814
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5312812, upper bound: 27.5318093
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5312812, upper bound: 27.5312812
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5312812, upper bound: 27.5312812
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5312812, upper bound: 27.5312812
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5314869
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5298837
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5149186, upper bound: 27.5149186
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5149186, upper bound: 27.5149186
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5149186, upper bound: 27.5149186
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5149186, upper bound: 27.5149186
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296920
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5149186, upper bound: 27.5149186
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5149186, upper bound: 27.5149186
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5324607, upper bound: 27.5372131
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5324607, upper bound: 27.5379062
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4998287, upper bound: 27.4998287
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4998287, upper bound: 27.4998287
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5183871, upper bound: 27.5183871
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5183871, upper bound: 27.5183871
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5304388
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5299140
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4989210, upper bound: 27.4989210
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4989210, upper bound: 27.4989219
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4989210, upper bound: 27.4989210
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4989210, upper bound: 27.4989219
time: 0.74 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.4989210, upper bound: 27.4989210
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.4989210, upper bound: 27.4989210
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5346178
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5341377
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5183871, upper bound: 27.5183871
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5183871, upper bound: 27.5183871
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.4989210, upper bound: 27.5017485
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.4989210, upper bound: 27.5017485
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.4889254, upper bound: 27.4889254
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.4889254, upper bound: 27.4889254
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5297302, upper bound: 27.5317432
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5297302, upper bound: 27.5297302
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.4902814, upper bound: 27.4902814
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.4902814, upper bound: 27.4902814
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.4902814, upper bound: 27.4902814
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.4902814, upper bound: 27.4902814
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5312812, upper bound: 27.5318093
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5312812, upper bound: 27.5312812
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5312812, upper bound: 27.5312812
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5312812, upper bound: 27.5312812
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5314869
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5298837
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5149186, upper bound: 27.5149186
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5149186, upper bound: 27.5149186
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5149186, upper bound: 27.5149186
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5149186, upper bound: 27.5149186
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296920
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5149186, upper bound: 27.5149186
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5149186, upper bound: 27.5149186
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5324607, upper bound: 27.5372131
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5324607, upper bound: 27.5379062
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.4998287, upper bound: 27.4998287
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.4998287, upper bound: 27.4998287
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5183871, upper bound: 27.5183871
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5183871, upper bound: 27.5183871
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5304388
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5299140
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.4989210, upper bound: 27.4989210
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.4989210, upper bound: 27.4989219
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.4989210, upper bound: 27.4989210
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 3, lower bound: -27.4989210, upper bound: 27.4989219

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4889254, upper bound: 27.4890928
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4889254, upper bound: 27.4890928
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5311340
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5148912
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5148912
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5068649, upper bound: 27.5054735
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5065226, upper bound: 27.5054735
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5054735, upper bound: 27.5054735
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5054735, upper bound: 27.5054735
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5180160, upper bound: 27.5180160
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5180160, upper bound: 27.5180160
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5180160, upper bound: 27.5183087
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5180160, upper bound: 27.5183087
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5150016
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5150029
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5148912
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5148912
time: 0.62 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.4889254, upper bound: 27.4890928
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.4889254, upper bound: 27.4890928
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5151636, upper bound: 27.5151636
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5311340
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5148912
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5148912
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5068649, upper bound: 27.5054735
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5065226, upper bound: 27.5054735
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5054735, upper bound: 27.5054735
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5054735, upper bound: 27.5054735
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5180160, upper bound: 27.5180160
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5180160, upper bound: 27.5180160
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5180160, upper bound: 27.5183087
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5180160, upper bound: 27.5183087
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5150016
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5150029
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5148912
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.94
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5148912

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.74 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.97 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
Binary search (step 2): status=Status.VERIFIED, low=0.1093750, high=0.1250000, mid=0.1093750, abs_max=31.572500228881836
rel_dist={3: [-27.547796646245764, 27.54779664624577]}

## Binary search (step 3) starts
Candidate diff: 0.1171875


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5455986, upper bound: 27.5455986
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5455986, upper bound: 27.5459277
time: 0.67 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.50 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.50
Output dim: 3, lower bound: -27.5455986, upper bound: 27.5455986
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.50
Output dim: 3, lower bound: -27.5455986, upper bound: 27.5459277

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5440054, upper bound: 27.5443087
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5440054, upper bound: 27.5451265
time: 0.74 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5408787, upper bound: 27.5455062
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5408787, upper bound: 27.5408787
time: 0.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.55 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 3, lower bound: -27.5440054, upper bound: 27.5443087
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 3, lower bound: -27.5440054, upper bound: 27.5451265
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 3, lower bound: -27.5408787, upper bound: 27.5455062
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 3, lower bound: -27.5408787, upper bound: 27.5408787

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5417230, upper bound: 27.5425186
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5421845, upper bound: 27.5425282
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5380681, upper bound: 27.5438072
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5437386, upper bound: 27.5446054
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5134498, upper bound: 27.5139044
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5134498, upper bound: 27.5139044
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5392923, upper bound: 27.5352708
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5352708, upper bound: 27.5392923
time: 0.84 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.84 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 3, lower bound: -27.5417230, upper bound: 27.5425186
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 3, lower bound: -27.5421845, upper bound: 27.5425282
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 3, lower bound: -27.5380681, upper bound: 27.5438072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 3, lower bound: -27.5437386, upper bound: 27.5446054
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.84
Output dim: 3, lower bound: -27.5134498, upper bound: 27.5139044
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.84
Output dim: 3, lower bound: -27.5134498, upper bound: 27.5139044
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 3, lower bound: -27.5392923, upper bound: 27.5352708
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 3, lower bound: -27.5352708, upper bound: 27.5392923

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5184184, upper bound: 27.5184184
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5184477, upper bound: 27.5184184
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5121934, upper bound: 27.5121934
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5131874, upper bound: 27.5121934
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5361357, upper bound: 27.5419883
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5361722, upper bound: 27.5415402
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5352475, upper bound: 27.5431776
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5410282, upper bound: 27.5430975
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5128039, upper bound: 27.5128039
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5128039, upper bound: 27.5128039
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5415298, upper bound: 27.5368897
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5336287, upper bound: 27.5371986
time: 0.74 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5184184, upper bound: 27.5184184
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5184477, upper bound: 27.5184184
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5121934, upper bound: 27.5121934
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5131874, upper bound: 27.5121934
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5361357, upper bound: 27.5419883
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5361722, upper bound: 27.5415402
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5352475, upper bound: 27.5431776
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5410282, upper bound: 27.5430975
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5128039, upper bound: 27.5128039
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5128039, upper bound: 27.5128039
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5415298, upper bound: 27.5368897
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 3, lower bound: -27.5336287, upper bound: 27.5371986

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5334949, upper bound: 27.5395671
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5345704, upper bound: 27.5333459
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5316680, upper bound: 27.5392344
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5316680, upper bound: 27.5383999
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5130911, upper bound: 27.5130911
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5130911, upper bound: 27.5130911
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5370821, upper bound: 27.5408794
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5327086, upper bound: 27.5358738
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5413802, upper bound: 27.5332162
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5331973, upper bound: 27.5367653
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5331419, upper bound: 27.5371323
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5331419, upper bound: 27.5334112
time: 0.73 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.68 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 3, lower bound: -27.5334949, upper bound: 27.5395671
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 3, lower bound: -27.5345704, upper bound: 27.5333459
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 3, lower bound: -27.5316680, upper bound: 27.5392344
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 3, lower bound: -27.5316680, upper bound: 27.5383999
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.68
Output dim: 3, lower bound: -27.5130911, upper bound: 27.5130911
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.68
Output dim: 3, lower bound: -27.5130911, upper bound: 27.5130911
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 3, lower bound: -27.5370821, upper bound: 27.5408794
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 3, lower bound: -27.5327086, upper bound: 27.5358738
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 3, lower bound: -27.5413802, upper bound: 27.5332162
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 3, lower bound: -27.5331973, upper bound: 27.5367653
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 3, lower bound: -27.5331419, upper bound: 27.5371323
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 3, lower bound: -27.5331419, upper bound: 27.5334112

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5329855, upper bound: 27.5392911
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5331453, upper bound: 27.5329855
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5300845, upper bound: 27.5300845
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5316313, upper bound: 27.5300845
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5311838, upper bound: 27.5386631
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5311838, upper bound: 27.5311838
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5149535, upper bound: 27.5149535
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5149535, upper bound: 27.5149535
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5300845, upper bound: 27.5383775
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5300845, upper bound: 27.5387341
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5321917, upper bound: 27.5353592
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5321917, upper bound: 27.5337743
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5409321, upper bound: 27.5329855
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5389030, upper bound: 27.5329997
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5302984, upper bound: 27.5343770
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5302984, upper bound: 27.5305962
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5302753, upper bound: 27.5349091
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5302753, upper bound: 27.5313352
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4988936, upper bound: 27.4988936
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4988936, upper bound: 27.4988936
time: 0.89 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.18 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -27.5329855, upper bound: 27.5392911
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -27.5331453, upper bound: 27.5329855
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -27.5300845, upper bound: 27.5300845
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -27.5316313, upper bound: 27.5300845
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -27.5311838, upper bound: 27.5386631
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -27.5311838, upper bound: 27.5311838
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -27.5149535, upper bound: 27.5149535
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -27.5149535, upper bound: 27.5149535
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -27.5300845, upper bound: 27.5383775
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -27.5300845, upper bound: 27.5387341
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -27.5321917, upper bound: 27.5353592
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -27.5321917, upper bound: 27.5337743
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -27.5409321, upper bound: 27.5329855
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -27.5389030, upper bound: 27.5329997
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -27.5302984, upper bound: 27.5343770
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -27.5302984, upper bound: 27.5305962
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -27.5302753, upper bound: 27.5349091
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -27.5302753, upper bound: 27.5313352
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -27.4988936, upper bound: 27.4988936
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -27.4988936, upper bound: 27.4988936

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5324607, upper bound: 27.5367187
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5324607, upper bound: 27.5324607
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5324607, upper bound: 27.5324607
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5326492, upper bound: 27.5324607
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5055122, upper bound: 27.5055122
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5055122, upper bound: 27.5055122
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5149186, upper bound: 27.5149186
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5149186, upper bound: 27.5149186
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5355766
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5055122, upper bound: 27.5055122
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5055122, upper bound: 27.5055122
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5356420
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5299888
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5318814, upper bound: 27.5352927
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5318814, upper bound: 27.5318814
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5301479
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5310372
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5300905, upper bound: 27.5343533
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5300905, upper bound: 27.5305615
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5146016, upper bound: 27.5146016
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5146016, upper bound: 27.5146016
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5297302, upper bound: 27.5297302
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5297302, upper bound: 27.5340472
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
time: 0.64 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.75 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5324607, upper bound: 27.5367187
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5324607, upper bound: 27.5324607
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5324607, upper bound: 27.5324607
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5326492, upper bound: 27.5324607
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5055122, upper bound: 27.5055122
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5055122, upper bound: 27.5055122
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5149186, upper bound: 27.5149186
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5149186, upper bound: 27.5149186
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5355766
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5055122, upper bound: 27.5055122
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5055122, upper bound: 27.5055122
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5356420
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5298837, upper bound: 27.5299888
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5318814, upper bound: 27.5352927
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5318814, upper bound: 27.5318814
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5301479
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5310372
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5296492, upper bound: 27.5296492
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5300905, upper bound: 27.5343533
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5300905, upper bound: 27.5305615
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5146016, upper bound: 27.5146016
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5146016, upper bound: 27.5146016
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5297302, upper bound: 27.5297302
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5297302, upper bound: 27.5340472
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 3, lower bound: -27.5145719, upper bound: 27.5145719

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5343591
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5338651
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4988936, upper bound: 27.4988936
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4988936, upper bound: 27.4988936
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4988936, upper bound: 27.4988936
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4988936, upper bound: 27.4988936
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4988936, upper bound: 27.4988936
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4988936, upper bound: 27.4988936
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5339978
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5294911, upper bound: 27.5294911
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4852876, upper bound: 27.4852876
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5148912
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5148912, upper bound: 27.5148912
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706
1: -6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965
2: -5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000
3: -7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002
4: -5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062

Time for backsubstitution: 2.48 seconds
Binary search (step 3): status=Status.UNKNOWN, low=0.1093750, high=0.1171875, mid=0.1171875, abs_max=31.572500228881836
rel_dist={3: [-27.547796646245764, 27.54779664624577]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.109375
execution time: 1123.93 seconds
