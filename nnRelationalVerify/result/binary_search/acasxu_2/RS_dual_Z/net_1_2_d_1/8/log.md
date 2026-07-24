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
execution time: IAR + LP analysis = 2.05 + 1.48 = 3.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -43.6007569, upper bound: 43.6007569


# Binary Search by BASE starts (time budget: 1196.47 seconds, max iter: 100)

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
Binary search time: 62.52 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1133.95 seconds

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.52 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.25 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1

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
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.51 seconds

## BFS RS instance: RS_RSZ2

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
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.51 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.20 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.56 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.19 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.19
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3723444
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.53 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3687788
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.59 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.65
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.65
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.65
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.65
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.65
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.65
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.65
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.65
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3820757, upper bound: 43.3642841
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
time: 0.53 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 3, lower bound: -43.3820757, upper bound: 43.3642841
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.62
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
Binary search (step 0): status=Status.VERIFIED, low=0.0625000, high=0.1250000, mid=0.0625000, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 1) starts
Candidate diff: 0.0937500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.48 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.14 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.14
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.14
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1

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
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.56 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.55 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3779762
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3779762
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.56 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.20 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3779762
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3723444
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.58 seconds

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
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.65 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.36 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.36
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.36
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.36
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.36
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.36
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.36
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.36
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.36
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.36
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3687788
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.59 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.60 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.60
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.60
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.60
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.60
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.60
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.60
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.60
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3657024, upper bound: 43.3642841
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3694196, upper bound: 43.3642841
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3820757, upper bound: 43.3642841
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3820757, upper bound: 43.3642841
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
time: 0.52 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.57 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.57
Output dim: 3, lower bound: -43.3657024, upper bound: 43.3642841
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.57
Output dim: 3, lower bound: -43.3694196, upper bound: 43.3642841
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.57
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.57
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.57
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.57
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.57
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.57
Output dim: 3, lower bound: -43.3820757, upper bound: 43.3642841
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.57
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.57
Output dim: 3, lower bound: -43.3820757, upper bound: 43.3642841
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.57
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.57
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.57
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.57
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
Binary search (step 1): status=Status.VERIFIED, low=0.0937500, high=0.1250000, mid=0.0937500, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 2) starts
Candidate diff: 0.1093750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.50 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.18 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.49 seconds

## BFS RS instance: RS_RSZ2

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
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.49 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.52 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.69 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.40 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.40
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.40
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.55 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.18 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.18
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.18
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.18
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.18
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.18
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.18
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.64 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.84 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.84
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.84
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.84
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.84
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.84
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.84
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3820757, upper bound: 43.3642841
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
time: 0.61 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.65 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 3, lower bound: -43.3820757, upper bound: 43.3642841
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.65
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
Binary search (step 2): status=Status.VERIFIED, low=0.1093750, high=0.1250000, mid=0.1093750, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157076]}

## Binary search (step 3) starts
Candidate diff: 0.1171875


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.56 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.29 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.29
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.29
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.49 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.48 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.27 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.27
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.27
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.27
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.27
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.71 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.33 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.33
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3723444
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.53 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.15 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.15
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.15
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.15
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.15
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.15
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.15
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.15
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.66 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.85 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.85
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.85
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.85
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.85
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.85
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.85
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.85
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.85
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.85
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
time: 0.63 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.69 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.69
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
Binary search (step 3): status=Status.VERIFIED, low=0.1171875, high=0.1250000, mid=0.1171875, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 4) starts
Candidate diff: 0.1210938


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.46 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.10 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.51 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.49 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.26 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.26
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.26
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.26
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.26
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.79 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.64 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.64
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.64
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.52 seconds

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
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.60 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.45 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.45
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.45
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.45
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.45
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.45
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.45
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 1.15 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.34 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.34
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.34
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.34
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.34
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.34
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.34
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.34
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3820757, upper bound: 43.3642841
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
time: 0.62 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.66 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3820757, upper bound: 43.3642841
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
Binary search (step 4): status=Status.VERIFIED, low=0.1210938, high=0.1250000, mid=0.1210938, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 5) starts
Candidate diff: 0.1230469


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.20 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.20
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.20
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.50 seconds

## BFS RS instance: RS_RSZ2

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
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.50 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.30 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.72 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.52 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.52
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.52
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.52
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.52
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.52
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.52
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.52
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.52
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.55 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.21 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.57 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.68 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.68
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.68
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.68
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.68
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.68
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.68
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3820757, upper bound: 43.3642841
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
time: 0.62 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.66 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3820757, upper bound: 43.3642841
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
Binary search (step 5): status=Status.VERIFIED, low=0.1230469, high=0.1250000, mid=0.1230469, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157076]}

## Binary search (step 6) starts
Candidate diff: 0.1240234


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.47 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.12 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.12
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.12
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1

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
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.50 seconds

## BFS RS instance: RS_RSZ2

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
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.50 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.38 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.79 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.57 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.57
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3723444
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3723444
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.54 seconds

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
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.52 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3687788
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3687788
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 1.24 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.63
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3687788
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.63
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.63
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.63
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.63
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.63
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.63
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.63
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.63
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.63
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3657024, upper bound: 43.3642841
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3694196, upper bound: 43.3642841
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
time: 0.64 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.79
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.79
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3642841
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.79
Output dim: 3, lower bound: -43.3657024, upper bound: 43.3642841
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.79
Output dim: 3, lower bound: -43.3694196, upper bound: 43.3642841
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.79
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.79
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.79
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.79
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.79
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.79
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.79
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.79
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
Binary search (step 6): status=Status.VERIFIED, low=0.1240234, high=0.1250000, mid=0.1240234, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 7) starts
Candidate diff: 0.1245117


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.22 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.22
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.22
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.51 seconds

## BFS RS instance: RS_RSZ2

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
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.49 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.80 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.75 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3723444
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3723444
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.56 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3687788
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3687788
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.58 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.82
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3687788
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.82
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.82
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.82
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.82
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.82
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.82
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.82
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.82
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.82
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3761216, upper bound: 43.3642841
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3762458, upper bound: 43.3642841
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3657024, upper bound: 43.3642841
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3694196, upper bound: 43.3642841
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
time: 0.64 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 3, lower bound: -43.3761216, upper bound: 43.3642841
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 3, lower bound: -43.3762458, upper bound: 43.3642841
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 3, lower bound: -43.3657024, upper bound: 43.3642841
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 3, lower bound: -43.3694196, upper bound: 43.3642841
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.77
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
Binary search (step 7): status=Status.VERIFIED, low=0.1245117, high=0.1250000, mid=0.1245117, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 8) starts
Candidate diff: 0.1247559


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.52 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.22 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.22
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.22
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1

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
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.51 seconds

## BFS RS instance: RS_RSZ2

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
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.51 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.35 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.84 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.84 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3723444
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.56 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.86 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.10 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.10
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.10
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.10
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.10
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.10
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.10
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.10
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.10
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3657024, upper bound: 43.3642841
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3694196, upper bound: 43.3642841
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3820757, upper bound: 43.3642841
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
time: 0.64 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.80 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 3, lower bound: -43.3657024, upper bound: 43.3642841
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 3, lower bound: -43.3694196, upper bound: 43.3642841
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 3, lower bound: -43.3820757, upper bound: 43.3642841
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
Binary search (step 8): status=Status.VERIFIED, low=0.1247559, high=0.1250000, mid=0.1247559, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 9) starts
Candidate diff: 0.1248779


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.52 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.23 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.23
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.23
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1

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
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.50 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.50 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.29 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.82 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.77 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.77
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.77
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.77
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.77
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.77
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.77
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.77
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.77
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3723444
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3723444
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.53 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3687788
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3687788
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.54 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.78 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.78
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3687788
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.78
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.78
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.78
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.78
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.78
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.78
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.78
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.78
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3761216, upper bound: 43.3642841
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3762458, upper bound: 43.3642841
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3657024, upper bound: 43.3642841
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3694196, upper bound: 43.3642841
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3820757, upper bound: 43.3642841
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
time: 0.65 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.85 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.85
Output dim: 3, lower bound: -43.3761216, upper bound: 43.3642841
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.85
Output dim: 3, lower bound: -43.3762458, upper bound: 43.3642841
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.85
Output dim: 3, lower bound: -43.3657024, upper bound: 43.3642841
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.85
Output dim: 3, lower bound: -43.3694196, upper bound: 43.3642841
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.85
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.85
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.85
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.85
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.85
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.85
Output dim: 3, lower bound: -43.3820757, upper bound: 43.3642841
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.85
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.85
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.85
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.85
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
Binary search (step 9): status=Status.VERIFIED, low=0.1248779, high=0.1250000, mid=0.1248779, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 10) starts
Candidate diff: 0.1249390


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.22 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.22
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.22
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1

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
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.50 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.49 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.26 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.26
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.26
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.26
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.26
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.74 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.53 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 3, lower bound: -43.3843393, upper bound: 43.3779762
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.53
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.56 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3723444
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.54 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.76 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.76
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.76
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.76
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.76
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.76
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.76
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.76
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3820757, upper bound: 43.3642841
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
time: 0.71 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.82 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 3, lower bound: -43.3819968, upper bound: 43.3642841
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 3, lower bound: -43.3820757, upper bound: 43.3642841
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3694196
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3657024
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3762458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3761216
Binary search (step 10): status=Status.VERIFIED, low=0.1249390, high=0.1250000, mid=0.1249390, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 11) starts
Candidate diff: 0.1249695


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.21 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.21
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.21
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1

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
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.49 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
time: 0.49 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.20 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5698562
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.20
Output dim: 3, lower bound: -43.5698562, upper bound: 43.5701711

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
time: 0.89 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.74 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.74
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.74
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3779762
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
time: 0.52 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 3, lower bound: -43.3843033, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 3, lower bound: -43.3779762, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3843393

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.61 seconds

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788
time: 0.61 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.99 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.99
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843033
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.99
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.99
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.99
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.99
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3687788
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -43.3687788, upper bound: 43.3843393
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.99
Output dim: 3, lower bound: -43.3723444, upper bound: 43.3687788

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3820757
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3642841, upper bound: 43.3819968
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307
1: -8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275
2: -9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216
3: -14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580
4: -14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412

Time for backsubstitution: 1.99 seconds
Binary search (step 11): status=Status.UNKNOWN, low=0.1249390, high=0.1249695, mid=0.1249695, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.12493896484375
execution time: 1134.00 seconds
