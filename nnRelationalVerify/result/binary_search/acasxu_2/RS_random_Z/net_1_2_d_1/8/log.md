## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_2.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 43.3827531155


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307)
1: (-8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275)
2: (-9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216)
3: (-14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580)
4: (-14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412)

## BASE Result
execution time: IAR + LP analysis = 2.04 + 1.51 = 3.55 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -43.6007569, upper bound: 43.6007569


# Binary Search by BASE starts (time budget: 1196.45 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 3) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 4) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=46.890357971191406
rel_dist={3: [-43.60068373509955, 43.60068373509955]}

## Binary search (step 5) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=46.890357971191406
rel_dist={3: [-43.6005733842848, 43.6005733842848]}

## Binary search (step 6) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=46.890357971191406
rel_dist={3: [-43.60049262479107, 43.60049262479107]}

## Binary search (step 7) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=46.890357971191406
rel_dist={3: [-43.600442096761945, 43.60044209676194]}

## Binary search (step 8) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=46.890357971191406
rel_dist={3: [-43.600415482278315, 43.60041548227832]}

## Binary search (step 9) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=46.890357971191406
rel_dist={3: [-43.60040197566741, 43.60040197566741]}

## Binary search (step 10) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=46.890357971191406
rel_dist={3: [-43.600395222369386, 43.600395222369386]}

## Binary search (step 11) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=46.890357971191406
rel_dist={3: [-43.600391845735146, 43.600391845735146]}

## Binary search (step 12) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=46.890357971191406
rel_dist={3: [-43.60039015744708, 43.60039015744708]}

## Binary search (step 13) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=46.890357971191406
rel_dist={3: [-43.60038931335932, 43.60038931335933]}

## Binary search (step 14) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=46.890357971191406
rel_dist={3: [-43.60038889142116, 43.60038889142115]}

## Binary search (step 15) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=46.890357971191406
rel_dist={3: [-43.60038868063948, 43.60038868930823]}

## Binary search (step 16) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=46.890357971191406
rel_dist={3: [-43.60038857723265, 43.60038857813507]}

## Binary Search Result
Binary search time: 62.87 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1133.58 seconds

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5839320, upper bound: 43.5839320
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5839320, upper bound: 43.5839320
time: 0.48 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.00 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.00
Output dim: 3, lower bound: -43.5839320, upper bound: 43.5839320
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.00
Output dim: 3, lower bound: -43.5839320, upper bound: 43.5839320

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
time: 0.49 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5809501, upper bound: 43.5803967
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5809501, upper bound: 43.5807978
time: 0.53 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.94 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 3, lower bound: -43.5809501, upper bound: 43.5803967
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 3, lower bound: -43.5809501, upper bound: 43.5807978

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3687788
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5802966, upper bound: 43.5802955
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5802966, upper bound: 43.5803094
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.52 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.93 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.93
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 3, lower bound: -43.5802966, upper bound: 43.5802955
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 3, lower bound: -43.5802966, upper bound: 43.5803094
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3800298, upper bound: 43.3675017
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3675017, upper bound: 43.3675017
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3685793, upper bound: 43.3710682
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3685793, upper bound: 43.3777123
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3685793, upper bound: 43.3685793
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3685793, upper bound: 43.3685793
time: 1.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5708890, upper bound: 43.5710844
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5709400, upper bound: 43.5710844
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5622385, upper bound: 43.5622451
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5622385, upper bound: 43.5622451
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.55 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.01 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 3, lower bound: -43.3800298, upper bound: 43.3675017
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 3, lower bound: -43.3675017, upper bound: 43.3675017
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 3, lower bound: -43.3685793, upper bound: 43.3710682
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 3, lower bound: -43.3685793, upper bound: 43.3777123
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 3, lower bound: -43.3685793, upper bound: 43.3685793
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 3, lower bound: -43.3685793, upper bound: 43.3685793
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 3, lower bound: -43.5708890, upper bound: 43.5710844
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 3, lower bound: -43.5709400, upper bound: 43.5710844
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 3, lower bound: -43.5622385, upper bound: 43.5622451
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 3, lower bound: -43.5622385, upper bound: 43.5622451
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5250554, upper bound: 43.5250554
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5250554, upper bound: 43.5250554
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5250554, upper bound: 43.5250554
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5250554, upper bound: 43.5250554
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3675017, upper bound: 43.3675017
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3675017, upper bound: 43.3800298
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3702141, upper bound: 43.3694196
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.49 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.97 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -43.5250554, upper bound: 43.5250554
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -43.5250554, upper bound: 43.5250554
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -43.5250554, upper bound: 43.5250554
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -43.5250554, upper bound: 43.5250554
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.97
Output dim: 3, lower bound: -43.3675017, upper bound: 43.3675017
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.97
Output dim: 3, lower bound: -43.3675017, upper bound: 43.3800298
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.97
Output dim: 3, lower bound: -43.3702141, upper bound: 43.3694196
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.97
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5490724
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5490724
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5235389, upper bound: 43.5235389
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5235389, upper bound: 43.5235389
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5235389, upper bound: 43.5235389
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5235389, upper bound: 43.5235389
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4678791, upper bound: 43.4678791
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4678791, upper bound: 43.4678791
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3685793, upper bound: 43.3685793
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3685793, upper bound: 43.3840494
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
time: 0.53 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.09 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5488756
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5490724
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5490724
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5493297
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 3, lower bound: -43.5235389, upper bound: 43.5235389
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 3, lower bound: -43.5235389, upper bound: 43.5235389
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 3, lower bound: -43.5235389, upper bound: 43.5235389
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 3, lower bound: -43.5235389, upper bound: 43.5235389
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 3, lower bound: -43.4678791, upper bound: 43.4678791
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 3, lower bound: -43.4678791, upper bound: 43.4678791
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.09
Output dim: 3, lower bound: -43.3685793, upper bound: 43.3685793
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 3, lower bound: -43.3685793, upper bound: 43.3840494
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.09
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.09
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 1.69 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477822
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477970
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 1.74 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5472474
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5473989
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 17

Time for candidate selection: 1.70 seconds

### Candidate
type: RSZ, layer: 3, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477275
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477275
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 1.72 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5472761
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5473403
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 1.67 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5470284
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5472530
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 41

Time for candidate selection: 1.67 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5473476
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5474541
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 1.67 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5490724
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5489807
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 16

Time for candidate selection: 1.68 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5474790
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5474791
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4678791, upper bound: 43.4678791
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4678791, upper bound: 43.4678791
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3673198, upper bound: 43.3673198
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3673198, upper bound: 43.3673198
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3673198, upper bound: 43.3673198
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3673198, upper bound: 43.3673198
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3673198, upper bound: 43.3673198
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3673198, upper bound: 43.3673198
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3673198, upper bound: 43.3673198
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3673198, upper bound: 43.3798200
time: 0.51 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.54 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477822
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477970
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5472474
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5473989
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477275
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477275
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5472761
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5473403
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5470284
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.5469905, upper bound: 43.5472530
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5473476
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5474541
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5490724
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.5488756, upper bound: 43.5489807
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5474790
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5474791
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.4678791, upper bound: 43.4678791
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.4678791, upper bound: 43.4678791
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.3673198, upper bound: 43.3673198
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.3673198, upper bound: 43.3673198
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.3673198, upper bound: 43.3673198
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.3673198, upper bound: 43.3673198
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.3673198, upper bound: 43.3673198
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.3673198, upper bound: 43.3673198
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.3673198, upper bound: 43.3673198
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 3, lower bound: -43.3673198, upper bound: 43.3798200

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477408
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477822
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5458251, upper bound: 43.5458331
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5458251, upper bound: 43.5458816
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5470926
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5472474
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5453954
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5454662
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5458251, upper bound: 43.5458251
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5458251, upper bound: 43.5458251
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5455585, upper bound: 43.5455585
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5455585, upper bound: 43.5455585
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5459173
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5457234
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5471011
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5473403
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452065, upper bound: 43.5452065
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452065, upper bound: 43.5452065
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452065, upper bound: 43.5452148
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452065, upper bound: 43.5453778
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5473269
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5473476
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5460897
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5460889
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5478391
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5478391
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5471011
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5471173
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5472964
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5474790
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5472857
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5474791
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3673198, upper bound: 43.3673198
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3673198, upper bound: 43.3673198
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.50 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.12 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477408
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5477822
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5458251, upper bound: 43.5458331
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5458251, upper bound: 43.5458816
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5470926
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5472474
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5453954
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5454662
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5458251, upper bound: 43.5458251
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5458251, upper bound: 43.5458251
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5455585, upper bound: 43.5455585
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5455585, upper bound: 43.5455585
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5459173
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5457234
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5471011
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5473403
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5452065, upper bound: 43.5452065
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5452065, upper bound: 43.5452065
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5452065, upper bound: 43.5452148
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5452065, upper bound: 43.5453778
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5473269
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5473476
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5460897
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5460889
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5478391
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5477275, upper bound: 43.5478391
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5471011
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5471173
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5472964
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5474790
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5472857
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5474791
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.3673198, upper bound: 43.3673198
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.3673198, upper bound: 43.3673198
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5425828, upper bound: 43.5425828
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5425828, upper bound: 43.5425828
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5425828, upper bound: 43.5425828
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5425828, upper bound: 43.5425828
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5436872, upper bound: 43.5437418
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5436872, upper bound: 43.5436872
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437411, upper bound: 43.5437411
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437411, upper bound: 43.5437411
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5470926
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5470926
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457250, upper bound: 43.5459725
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457250, upper bound: 43.5459469
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5452805
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5453954
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5391195, upper bound: 43.5391195
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5391195, upper bound: 43.5398326
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437411, upper bound: 43.5437411
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437411, upper bound: 43.5437411
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5458251, upper bound: 43.5458251
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5458251, upper bound: 43.5458251
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5403234, upper bound: 43.5403234
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5403234, upper bound: 43.5403234
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5403234, upper bound: 43.5403234
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5403234, upper bound: 43.5403234
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5438330, upper bound: 43.5438905
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5438330, upper bound: 43.5439806
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5456908
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5457234
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5410170, upper bound: 43.5410170
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5410170, upper bound: 43.5410170
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452065, upper bound: 43.5455531
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452065, upper bound: 43.5454791
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5438223, upper bound: 43.5439763
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5438223, upper bound: 43.5439763
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5391582, upper bound: 43.5391582
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5391582, upper bound: 43.5391582
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452065, upper bound: 43.5452148
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452065, upper bound: 43.5452098
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5438223, upper bound: 43.5438355
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5438223, upper bound: 43.5438355
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5456666
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5459711
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5471011
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5473476
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5420311, upper bound: 43.5420311
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5420311, upper bound: 43.5420311
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5459877
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5460889
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5455585, upper bound: 43.5455585
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457556, upper bound: 43.5455585
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5425828, upper bound: 43.5425828
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5425828, upper bound: 43.5425828
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5471011
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5471011
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452065, upper bound: 43.5452065
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452065, upper bound: 43.5452587
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Candidate
type: RSZ, layer: 3, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5472964
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5472947
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5454583
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5455812
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437411, upper bound: 43.5437411
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437411, upper bound: 43.5437411
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5456812
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5457088
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3578442, upper bound: 43.3572016
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3578030, upper bound: 43.3572016
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.57 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.27 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5425828, upper bound: 43.5425828
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5425828, upper bound: 43.5425828
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5425828, upper bound: 43.5425828
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5425828, upper bound: 43.5425828
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5436872, upper bound: 43.5437418
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5436872, upper bound: 43.5436872
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5437411, upper bound: 43.5437411
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5437411, upper bound: 43.5437411
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5470926
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5470926
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5457250, upper bound: 43.5459725
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5457250, upper bound: 43.5459469
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5452805
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5453954
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5391195, upper bound: 43.5391195
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5391195, upper bound: 43.5398326
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5437411, upper bound: 43.5437411
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5437411, upper bound: 43.5437411
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5458251, upper bound: 43.5458251
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5458251, upper bound: 43.5458251
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5403234, upper bound: 43.5403234
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5403234, upper bound: 43.5403234
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5403234, upper bound: 43.5403234
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5403234, upper bound: 43.5403234
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5438330, upper bound: 43.5438905
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5438330, upper bound: 43.5439806
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5456908
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5457234
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5410170, upper bound: 43.5410170
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5410170, upper bound: 43.5410170
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5452065, upper bound: 43.5455531
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5452065, upper bound: 43.5454791
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5438223, upper bound: 43.5439763
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5438223, upper bound: 43.5439763
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5391582, upper bound: 43.5391582
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5391582, upper bound: 43.5391582
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5452065, upper bound: 43.5452148
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5452065, upper bound: 43.5452098
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5438223, upper bound: 43.5438355
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5438223, upper bound: 43.5438355
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5456666
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5459711
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5471011
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5473476
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5420311, upper bound: 43.5420311
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5420311, upper bound: 43.5420311
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5459877
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5456657, upper bound: 43.5460889
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5455585, upper bound: 43.5455585
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5457556, upper bound: 43.5455585
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5425828, upper bound: 43.5425828
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5425828, upper bound: 43.5425828
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5471011
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5471011, upper bound: 43.5471011
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5452065, upper bound: 43.5452065
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5452065, upper bound: 43.5452587
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5472964
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5472947
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5454583
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5452805, upper bound: 43.5455812
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5437411, upper bound: 43.5437411
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5437411, upper bound: 43.5437411
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5456812
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.5452674, upper bound: 43.5457088
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3578442, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3578030, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5399400, upper bound: 43.5399400
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5399400, upper bound: 43.5399400
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5425828, upper bound: 43.5425828
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5425828, upper bound: 43.5425828
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5399400, upper bound: 43.5399400
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5399400, upper bound: 43.5399400
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5405406, upper bound: 43.5405406
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5405406, upper bound: 43.5405406
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5436872, upper bound: 43.5437182
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5436872, upper bound: 43.5437418
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5419732, upper bound: 43.5419732
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5419732, upper bound: 43.5419732
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5416433, upper bound: 43.5416433
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5416433, upper bound: 43.5416433
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5378462, upper bound: 43.5378462
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5378462, upper bound: 43.5378462
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457250, upper bound: 43.5457250
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457250, upper bound: 43.5457250
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Candidate
type: RSZ, layer: 3, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5470926
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5470926, upper bound: 43.5470926
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457250, upper bound: 43.5458695
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5457250, upper bound: 43.5459725
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.17 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5595530, upper bound: 43.5595530
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5595530, upper bound: 43.5595530
time: 0.52 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.06 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.06
Output dim: 3, lower bound: -43.5595530, upper bound: 43.5595530
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.06
Output dim: 3, lower bound: -43.5595530, upper bound: 43.5595530

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5125662, upper bound: 43.5125662
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5125662, upper bound: 43.5125662
time: 0.50 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5591323, upper bound: 43.5593314
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5591323, upper bound: 43.5591323
time: 0.56 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.00 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 3, lower bound: -43.5125662, upper bound: 43.5125662
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 3, lower bound: -43.5125662, upper bound: 43.5125662
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 3, lower bound: -43.5591323, upper bound: 43.5593314
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 3, lower bound: -43.5591323, upper bound: 43.5591323

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5125662, upper bound: 43.5125662
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5125662, upper bound: 43.5125662
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3685793, upper bound: 43.3685793
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3710682, upper bound: 43.3685793
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.52 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.91 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 3, lower bound: -43.5125662, upper bound: 43.5125662
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 3, lower bound: -43.5125662, upper bound: 43.5125662
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.91
Output dim: 3, lower bound: -43.3685793, upper bound: 43.3685793
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.91
Output dim: 3, lower bound: -43.3710682, upper bound: 43.3685793
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5125662, upper bound: 43.5125662
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5125662, upper bound: 43.5125662
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.47 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.83
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.83
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.83
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.83
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.83
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.83
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.83
Output dim: 3, lower bound: -43.5125662, upper bound: 43.5125662
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.83
Output dim: 3, lower bound: -43.5125662, upper bound: 43.5125662
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.83
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.83
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.83
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.83
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3716156
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3716156
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3680681
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3680681
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.48 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3716156
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3716156
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3680681
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3680681
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3698602
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3698602
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3653406
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3751452
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
time: 0.50 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3698602
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3698602
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3653406
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3751452
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.5123461, upper bound: 43.5123461
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.05
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3799496
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3799496
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3578030, upper bound: 43.3572016
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3680681, upper bound: 43.3639130
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3680681, upper bound: 43.3639130
time: 0.54 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3799496
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3799496
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.3578030, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.4830080, upper bound: 43.4830080
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.3680681, upper bound: 43.3639130
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 3, lower bound: -43.3680681, upper bound: 43.3639130

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3608085
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3671719
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3759396
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3759396
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
time: 0.79 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.09 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3608085
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3671719
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3759396
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3759396
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.09
Output dim: 3, lower bound: -43.3639130, upper bound: 43.3639130

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3578030
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3578442
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572985
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572985
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.57 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3578030
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3578442
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572985
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572985
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.27
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
Binary search (step 1): status=Status.VERIFIED, low=0.0312500, high=0.0625000, mid=0.0312500, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 2) starts
Candidate diff: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5635211, upper bound: 43.5635211
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5635211, upper bound: 43.5635211
time: 0.52 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.07 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.07
Output dim: 3, lower bound: -43.5635211, upper bound: 43.5635211
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.07
Output dim: 3, lower bound: -43.5635211, upper bound: 43.5635211

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5589663, upper bound: 43.5589663
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5589663, upper bound: 43.5589663
time: 0.54 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5634816, upper bound: 43.5634866
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5634816, upper bound: 43.5634816
time: 0.47 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.93 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 3, lower bound: -43.5589663, upper bound: 43.5589663
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 3, lower bound: -43.5589663, upper bound: 43.5589663
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 3, lower bound: -43.5634816, upper bound: 43.5634866
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 3, lower bound: -43.5634816, upper bound: 43.5634816

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3675017, upper bound: 43.3675017
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3675017, upper bound: 43.3675017
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5238452, upper bound: 43.5236602
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5238452, upper bound: 43.5236602
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5254368, upper bound: 43.5252159
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5252159, upper bound: 43.5252159
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5252159, upper bound: 43.5254368
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5252159, upper bound: 43.5254368
time: 0.53 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.02 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 3, lower bound: -43.3675017, upper bound: 43.3675017
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 3, lower bound: -43.3675017, upper bound: 43.3675017
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 3, lower bound: -43.5238452, upper bound: 43.5236602
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 3, lower bound: -43.5238452, upper bound: 43.5236602
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 3, lower bound: -43.5254368, upper bound: 43.5252159
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 3, lower bound: -43.5252159, upper bound: 43.5252159
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 3, lower bound: -43.5252159, upper bound: 43.5254368
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 3, lower bound: -43.5252159, upper bound: 43.5254368

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5220938, upper bound: 43.5220938
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5220938, upper bound: 43.5220938
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3715658
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3715658
time: 0.55 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.99 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -43.5220938, upper bound: 43.5220938
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -43.5220938, upper bound: 43.5220938
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3715658
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3715658

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
time: 0.47 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.36 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.36
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.36
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3606078, upper bound: 43.3572016
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572985
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572985
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3715658
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3715658
time: 0.56 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.22 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.3606078, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.5211636, upper bound: 43.5211636
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572985
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572985
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3715658
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.22
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3715658

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3634773
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3590294
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
time: 0.54 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.56 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4459505, upper bound: 43.4459505
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3634773
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3590294
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 3, lower bound: -43.5133142, upper bound: 43.5133142

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3621673, upper bound: 43.3572016
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3621673, upper bound: 43.3572016
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3572016, upper bound: 43.3572016
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3575275, upper bound: 43.3575275
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4457501, upper bound: 43.4457501
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.11 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0312500, high=0.0468750, mid=0.0468750, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.03125
execution time: 1135.09 seconds
