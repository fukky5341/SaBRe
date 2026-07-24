## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_3.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 4551.82301297592


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953)
1: (-2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344)
2: (-3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344)
3: (-1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836)
4: (-3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328)

## BASE Result
execution time: IAR + LP analysis = 2.03 + 2.14 = 4.18 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -4552.2890012, upper bound: 4552.2890012


# Binary Search by BASE starts (time budget: 1195.82 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=5687.5751953125
rel_dist={0: [-4552.2878636421, 4552.287863642101]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=5687.5751953125
rel_dist={0: [-4552.283859528013, 4552.283859528014]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=5687.5751953125
rel_dist={0: [-4552.279487443704, 4552.279487443706]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=5687.5751953125
rel_dist={0: [-4552.274177015944, 4552.274177015945]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=5687.5751953125
rel_dist={0: [-4552.269705383769, 4552.269705383769]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=5687.5751953125
rel_dist={0: [-4552.266690988818, 4552.266690988818]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=5687.5751953125
rel_dist={0: [-4552.265148835347, 4552.265148835348]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=5687.5751953125
rel_dist={0: [-4552.264371856103, 4552.264371856105]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=5687.5751953125
rel_dist={0: [-4552.263982959417, 4552.263982959417]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=5687.5751953125
rel_dist={0: [-4552.263788511317, 4552.263788511316]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=5687.5751953125
rel_dist={0: [-4552.263681482938, 4552.263681482938]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=5687.5751953125
rel_dist={0: [-4552.26362479584, 4552.26362479584]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=5687.5751953125
rel_dist={0: [-4552.263594127799, 4552.263594127799]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=5687.5751953125
rel_dist={0: [-4552.263578672278, 4552.2635786722785]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=5687.5751953125
rel_dist={0: [-4552.263570946265, 4552.263570944802]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=5687.5751953125
rel_dist={0: [-4552.263567081615, 4552.2635670963]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=5687.5751953125
rel_dist={0: [-4552.263565151061, 4552.263565152603]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=5687.5751953125
rel_dist={0: [-4552.2635641832, 4552.263564185958]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=5687.5751953125
rel_dist={0: [-4552.263564553653, 4552.263563751329]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=5687.5751953125
rel_dist={0: [-4552.263576470345, 4552.26356620421]}

## Binary Search Result
Binary search time: 84.63 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1111.20 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2878626, upper bound: 4552.2736919
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2736919, upper bound: 4552.2878626
time: 6.10 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 6.99 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 6.99
Output dim: 0, lower bound: -4552.2878626, upper bound: 4552.2736919
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 6.99
Output dim: 0, lower bound: -4552.2736919, upper bound: 4552.2878626

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2872610, upper bound: 4552.2731599
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2872610, upper bound: 4552.2735688
time: 0.93 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1736061, upper bound: 4551.1775769
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1736061, upper bound: 4551.1775981
time: 0.74 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.35 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 0, lower bound: -4552.2872610, upper bound: 4552.2731599
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 0, lower bound: -4552.2872610, upper bound: 4552.2735688
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.35
Output dim: 0, lower bound: -4551.1736061, upper bound: 4551.1775769
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.35
Output dim: 0, lower bound: -4551.1736061, upper bound: 4551.1775981

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4538.7848351, upper bound: 4538.7848351
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4538.7848351, upper bound: 4538.7848351
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2801272, upper bound: 4552.2735688
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2832133, upper bound: 4552.2733479
time: 1.13 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.89 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.89
Output dim: 0, lower bound: -4538.7848351, upper bound: 4538.7848351
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.89
Output dim: 0, lower bound: -4538.7848351, upper bound: 4538.7848351
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 0, lower bound: -4552.2801272, upper bound: 4552.2735688
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 0, lower bound: -4552.2832133, upper bound: 4552.2733479

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2621042, upper bound: 4552.2554178
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2609412, upper bound: 4552.2550740
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8449236, upper bound: 4551.8387504
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8459305, upper bound: 4551.8389116
time: 0.88 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -4552.2621042, upper bound: 4552.2554178
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -4552.2609412, upper bound: 4552.2550740
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -4551.8449236, upper bound: 4551.8387504
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -4551.8459305, upper bound: 4551.8389116

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4544.7022387, upper bound: 4544.7021686
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4544.7022387, upper bound: 4544.7021686
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4548.6489176, upper bound: 4548.6489176
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4548.6489176, upper bound: 4548.6489176
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8423506, upper bound: 4551.8367768
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8425171, upper bound: 4551.8366206
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7533841, upper bound: 4551.7520088
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7533841, upper bound: 4551.7526794
time: 0.83 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.54 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.54
Output dim: 0, lower bound: -4544.7022387, upper bound: 4544.7021686
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.54
Output dim: 0, lower bound: -4544.7022387, upper bound: 4544.7021686
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.54
Output dim: 0, lower bound: -4548.6489176, upper bound: 4548.6489176
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.54
Output dim: 0, lower bound: -4548.6489176, upper bound: 4548.6489176
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 0, lower bound: -4551.8423506, upper bound: 4551.8367768
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.54
Output dim: 0, lower bound: -4551.8425171, upper bound: 4551.8366206
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.54
Output dim: 0, lower bound: -4551.7533841, upper bound: 4551.7520088
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.54
Output dim: 0, lower bound: -4551.7533841, upper bound: 4551.7526794

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8363158, upper bound: 4551.8363243
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8414697, upper bound: 4551.8363158
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1146508, upper bound: 4551.1033720
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1146508, upper bound: 4551.1033720
time: 0.92 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.73 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -4551.8363158, upper bound: 4551.8363243
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 0, lower bound: -4551.8414697, upper bound: 4551.8363158
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.73
Output dim: 0, lower bound: -4551.1146508, upper bound: 4551.1033720
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.73
Output dim: 0, lower bound: -4551.1146508, upper bound: 4551.1033720

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8356533, upper bound: 4551.8356619
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8356533, upper bound: 4551.8356533
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8351977, upper bound: 4551.8351473
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8406975, upper bound: 4551.8351473
time: 0.89 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.55 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 0, lower bound: -4551.8356533, upper bound: 4551.8356619
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 0, lower bound: -4551.8356533, upper bound: 4551.8356533
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 0, lower bound: -4551.8351977, upper bound: 4551.8351473
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 0, lower bound: -4551.8406975, upper bound: 4551.8351473

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4547.7928225, upper bound: 4547.7928225
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4547.7928225, upper bound: 4547.7928225
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.8033772, upper bound: 4551.8033772
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.8033772, upper bound: 4551.8033772
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.0959358, upper bound: 4551.0959006
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.0959358, upper bound: 4551.0959006
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1075998, upper bound: 4551.0958958
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1075998, upper bound: 4551.0958958
time: 0.77 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.39 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.39
Output dim: 0, lower bound: -4547.7928225, upper bound: 4547.7928225
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.39
Output dim: 0, lower bound: -4547.7928225, upper bound: 4547.7928225
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.39
Output dim: 0, lower bound: -4551.8033772, upper bound: 4551.8033772
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.39
Output dim: 0, lower bound: -4551.8033772, upper bound: 4551.8033772
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.39
Output dim: 0, lower bound: -4551.0959358, upper bound: 4551.0959006
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.39
Output dim: 0, lower bound: -4551.0959358, upper bound: 4551.0959006
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.39
Output dim: 0, lower bound: -4551.1075998, upper bound: 4551.0958958
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.39
Output dim: 0, lower bound: -4551.1075998, upper bound: 4551.0958958
Binary search (step 0): status=Status.VERIFIED, low=0.5000000, high=1.0000000, mid=0.5000000, abs_max=5687.5751953125
rel_dist={0: [-4552.2878636421, 4552.287863642101]}

## Binary search (step 1) starts
Candidate diff: 0.7500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0372213, upper bound: 4552.0367589
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0367589, upper bound: 4552.0372213
time: 0.87 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 0, lower bound: -4552.0372213, upper bound: 4552.0367589
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 0, lower bound: -4552.0367589, upper bound: 4552.0372213

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2016080, upper bound: 4551.2013170
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2016080, upper bound: 4551.2013175
time: 0.89 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9610772, upper bound: 4551.9601336
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9494467, upper bound: 4551.9615970
time: 0.82 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.49 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.49
Output dim: 0, lower bound: -4551.2016080, upper bound: 4551.2013170
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.49
Output dim: 0, lower bound: -4551.2016080, upper bound: 4551.2013175
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.49
Output dim: 0, lower bound: -4551.9610772, upper bound: 4551.9601336
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.49
Output dim: 0, lower bound: -4551.9494467, upper bound: 4551.9615970

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9488924, upper bound: 4551.9537717
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9596806, upper bound: 4551.9597773
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9001835, upper bound: 4551.9126390
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9001902, upper bound: 4551.9064776
time: 0.89 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.37 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 0, lower bound: -4551.9488924, upper bound: 4551.9537717
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 0, lower bound: -4551.9596806, upper bound: 4551.9597773
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 0, lower bound: -4551.9001835, upper bound: 4551.9126390
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 0, lower bound: -4551.9001902, upper bound: 4551.9064776

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9384346, upper bound: 4551.9423571
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9382205, upper bound: 4551.9398066
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.9278275, upper bound: 4550.9285280
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.9278275, upper bound: 4550.9285989
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8741908, upper bound: 4551.8841687
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8741702, upper bound: 4551.8818807
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.8617413, upper bound: 4550.8670466
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.8617413, upper bound: 4550.8670466
time: 0.73 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.58 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.58
Output dim: 0, lower bound: -4551.9384346, upper bound: 4551.9423571
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.58
Output dim: 0, lower bound: -4551.9382205, upper bound: 4551.9398066
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.58
Output dim: 0, lower bound: -4550.9278275, upper bound: 4550.9285280
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.58
Output dim: 0, lower bound: -4550.9278275, upper bound: 4550.9285989
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.58
Output dim: 0, lower bound: -4551.8741908, upper bound: 4551.8841687
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.58
Output dim: 0, lower bound: -4551.8741702, upper bound: 4551.8818807
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.58
Output dim: 0, lower bound: -4550.8617413, upper bound: 4550.8670466
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.58
Output dim: 0, lower bound: -4550.8617413, upper bound: 4550.8670466

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.9734883, upper bound: 4550.9754769
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.9734883, upper bound: 4550.9754769
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9381622, upper bound: 4551.9390077
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9382205, upper bound: 4551.9381742
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4548.2917890, upper bound: 4548.2918329
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4548.2917890, upper bound: 4548.3004713
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7150987, upper bound: 4551.7153536
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7150987, upper bound: 4551.7221650
time: 0.78 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.01 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 0, lower bound: -4550.9734883, upper bound: 4550.9754769
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 0, lower bound: -4550.9734883, upper bound: 4550.9754769
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -4551.9381622, upper bound: 4551.9390077
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 0, lower bound: -4551.9382205, upper bound: 4551.9381742
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 0, lower bound: -4548.2917890, upper bound: 4548.2918329
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 0, lower bound: -4548.2917890, upper bound: 4548.3004713
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 0, lower bound: -4551.7150987, upper bound: 4551.7153536
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 0, lower bound: -4551.7150987, upper bound: 4551.7221650

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9381622, upper bound: 4551.9382017
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9381622, upper bound: 4551.9390077
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9377658, upper bound: 4551.9377074
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9377074, upper bound: 4551.9377195
time: 0.89 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.16 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 0, lower bound: -4551.9381622, upper bound: 4551.9382017
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 0, lower bound: -4551.9381622, upper bound: 4551.9390077
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 0, lower bound: -4551.9377658, upper bound: 4551.9377074
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 0, lower bound: -4551.9377074, upper bound: 4551.9377195

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9380804, upper bound: 4551.9381084
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9380804, upper bound: 4551.9380804
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9377074, upper bound: 4551.9377074
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9377074, upper bound: 4551.9385566
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4521311, upper bound: 4551.4520743
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4520743, upper bound: 4551.4520743
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3615619, upper bound: 4551.3543682
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3615619, upper bound: 4551.3543829
time: 0.75 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.92 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -4551.9380804, upper bound: 4551.9381084
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -4551.9380804, upper bound: 4551.9380804
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -4551.9377074, upper bound: 4551.9377074
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -4551.9377074, upper bound: 4551.9385566
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.92
Output dim: 0, lower bound: -4551.4521311, upper bound: 4551.4520743
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.92
Output dim: 0, lower bound: -4551.4520743, upper bound: 4551.4520743
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.92
Output dim: 0, lower bound: -4551.3615619, upper bound: 4551.3543682
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.92
Output dim: 0, lower bound: -4551.3615619, upper bound: 4551.3543829

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3548468, upper bound: 4551.3547560
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3548468, upper bound: 4551.3547891
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3548530, upper bound: 4551.3547560
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3548530, upper bound: 4551.3547560
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8918461, upper bound: 4551.8918461
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8918461, upper bound: 4551.8918461
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.9151722, upper bound: 4550.9151805
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.9151722, upper bound: 4550.9151805
time: 0.88 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.66 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.66
Output dim: 0, lower bound: -4551.3548468, upper bound: 4551.3547560
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.66
Output dim: 0, lower bound: -4551.3548468, upper bound: 4551.3547891
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.66
Output dim: 0, lower bound: -4551.3548530, upper bound: 4551.3547560
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.66
Output dim: 0, lower bound: -4551.3548530, upper bound: 4551.3547560
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.66
Output dim: 0, lower bound: -4551.8918461, upper bound: 4551.8918461
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.66
Output dim: 0, lower bound: -4551.8918461, upper bound: 4551.8918461
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.66
Output dim: 0, lower bound: -4550.9151722, upper bound: 4550.9151805
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.66
Output dim: 0, lower bound: -4550.9151722, upper bound: 4550.9151805

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.8833391, upper bound: 4550.8833391
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.8833391, upper bound: 4550.8833391
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.8833391, upper bound: 4550.8833391
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.8833391, upper bound: 4550.8833391
time: 0.84 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 5.05 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 5.05
Output dim: 0, lower bound: -4550.8833391, upper bound: 4550.8833391
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 5.05
Output dim: 0, lower bound: -4550.8833391, upper bound: 4550.8833391
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 5.05
Output dim: 0, lower bound: -4550.8833391, upper bound: 4550.8833391
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 5.05
Output dim: 0, lower bound: -4550.8833391, upper bound: 4550.8833391
Binary search (step 1): status=Status.VERIFIED, low=0.7500000, high=1.0000000, mid=0.7500000, abs_max=5687.5751953125
rel_dist={0: [-4552.289001211694, 4552.289001211693]}

## Binary search (step 2) starts
Candidate diff: 0.8750000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2878392, upper bound: 4552.2870918
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2870918, upper bound: 4552.2878392
time: 0.84 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.64 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.64
Output dim: 0, lower bound: -4552.2878392, upper bound: 4552.2870918
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.64
Output dim: 0, lower bound: -4552.2870918, upper bound: 4552.2878392

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2682123, upper bound: 4552.2698533
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2682123, upper bound: 4552.2698533
time: 0.91 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2863577, upper bound: 4552.2748883
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2748798, upper bound: 4552.2871052
time: 0.94 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.82 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.82
Output dim: 0, lower bound: -4552.2682123, upper bound: 4552.2698533
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.82
Output dim: 0, lower bound: -4552.2682123, upper bound: 4552.2698533
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.82
Output dim: 0, lower bound: -4552.2863577, upper bound: 4552.2748883
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.82
Output dim: 0, lower bound: -4552.2748798, upper bound: 4552.2871052

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6098176, upper bound: 4551.6097956
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6098176, upper bound: 4551.6097956
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4543.0664396, upper bound: 4543.0664396
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4543.0664396, upper bound: 4543.0664396
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4540.0008550, upper bound: 4540.0018718
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4540.0008550, upper bound: 4540.0018718
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2726690, upper bound: 4552.2857321
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2726724, upper bound: 4552.2857091
time: 0.81 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.52 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.52
Output dim: 0, lower bound: -4551.6098176, upper bound: 4551.6097956
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.52
Output dim: 0, lower bound: -4551.6098176, upper bound: 4551.6097956
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.52
Output dim: 0, lower bound: -4543.0664396, upper bound: 4543.0664396
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.52
Output dim: 0, lower bound: -4543.0664396, upper bound: 4543.0664396
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.52
Output dim: 0, lower bound: -4540.0008550, upper bound: 4540.0018718
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.52
Output dim: 0, lower bound: -4540.0008550, upper bound: 4540.0018718
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.52
Output dim: 0, lower bound: -4552.2726690, upper bound: 4552.2857321
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.52
Output dim: 0, lower bound: -4552.2726724, upper bound: 4552.2857091

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1127742, upper bound: 4551.1146498
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1127742, upper bound: 4551.1146498
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1118002, upper bound: 4551.1148438
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1118002, upper bound: 4551.1148438
time: 0.75 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.44 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -4551.1127742, upper bound: 4551.1146498
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -4551.1127742, upper bound: 4551.1146498
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -4551.1118002, upper bound: 4551.1148438
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -4551.1118002, upper bound: 4551.1148438
Binary search (step 2): status=Status.VERIFIED, low=0.8750000, high=1.0000000, mid=0.8750000, abs_max=5687.5751953125
rel_dist={0: [-4552.289001211694, 4552.289001211693]}

## Binary search (step 3) starts
Candidate diff: 0.9375000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4185101, upper bound: 4551.4185101
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4185101, upper bound: 4551.4185101
time: 0.93 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.90 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.90
Output dim: 0, lower bound: -4551.4185101, upper bound: 4551.4185101
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.90
Output dim: 0, lower bound: -4551.4185101, upper bound: 4551.4185101
Binary search (step 3): status=Status.VERIFIED, low=0.9375000, high=1.0000000, mid=0.9375000, abs_max=5687.5751953125
rel_dist={0: [-4552.289001211693, 4552.289001211693]}

## Binary search (step 4) starts
Candidate diff: 0.9687500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2860667, upper bound: 4552.2860667
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2860667, upper bound: 4552.2890012
time: 0.75 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.78 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.78
Output dim: 0, lower bound: -4552.2860667, upper bound: 4552.2860667
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.78
Output dim: 0, lower bound: -4552.2860667, upper bound: 4552.2890012

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2881480, upper bound: 4552.2800874
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2884988, upper bound: 4552.2859142
time: 0.73 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1156710, upper bound: 4551.1168031
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1156710, upper bound: 4551.1168031
time: 0.81 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.56 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.56
Output dim: 0, lower bound: -4552.2881480, upper bound: 4552.2800874
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.56
Output dim: 0, lower bound: -4552.2884988, upper bound: 4552.2859142
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.56
Output dim: 0, lower bound: -4551.1156710, upper bound: 4551.1168031
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.56
Output dim: 0, lower bound: -4551.1156710, upper bound: 4551.1168031

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4548.8661414, upper bound: 4548.8675040
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4548.8661414, upper bound: 4548.8675040
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2874425, upper bound: 4552.2779315
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2811563, upper bound: 4552.2852606
time: 0.89 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.75 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.75
Output dim: 0, lower bound: -4548.8661414, upper bound: 4548.8675040
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.75
Output dim: 0, lower bound: -4548.8661414, upper bound: 4548.8675040
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 0, lower bound: -4552.2874425, upper bound: 4552.2779315
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 0, lower bound: -4552.2811563, upper bound: 4552.2852606

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2867086, upper bound: 4552.2747120
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2713606, upper bound: 4552.2771955
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2718766, upper bound: 4552.2764518
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2724539, upper bound: 4552.2767996
time: 1.00 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.84 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -4552.2867086, upper bound: 4552.2747120
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -4552.2713606, upper bound: 4552.2771955
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -4552.2718766, upper bound: 4552.2764518
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 0, lower bound: -4552.2724539, upper bound: 4552.2767996

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2808993, upper bound: 4552.2725279
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2852919, upper bound: 4552.2726630
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2713606, upper bound: 4552.2763980
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2713606, upper bound: 4552.2713722
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.0951473, upper bound: 4551.0970174
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.0951473, upper bound: 4551.0970174
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2520009, upper bound: 4552.2642605
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2579537, upper bound: 4552.2609679
time: 0.98 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.78 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -4552.2808993, upper bound: 4552.2725279
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -4552.2852919, upper bound: 4552.2726630
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -4552.2713606, upper bound: 4552.2763980
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -4552.2713606, upper bound: 4552.2713722
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.78
Output dim: 0, lower bound: -4551.0951473, upper bound: 4551.0970174
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.78
Output dim: 0, lower bound: -4551.0951473, upper bound: 4551.0970174
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -4552.2520009, upper bound: 4552.2642605
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -4552.2579537, upper bound: 4552.2609679

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2646966, upper bound: 4552.2582109
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2646966, upper bound: 4552.2598726
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3808522, upper bound: 4551.3743504
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3780998, upper bound: 4551.3743504
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4009070, upper bound: 4551.4009265
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4009070, upper bound: 4551.4009265
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2690723, upper bound: 4552.2690723
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2690723, upper bound: 4552.2690838
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1796248, upper bound: 4552.1801595
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1797752, upper bound: 4552.1796773
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2941722, upper bound: 4551.2889280
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2941722, upper bound: 4551.2891986
time: 0.93 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.70 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -4552.2646966, upper bound: 4552.2582109
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -4552.2646966, upper bound: 4552.2598726
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 0, lower bound: -4551.3808522, upper bound: 4551.3743504
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 0, lower bound: -4551.3780998, upper bound: 4551.3743504
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 0, lower bound: -4551.4009070, upper bound: 4551.4009265
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 0, lower bound: -4551.4009070, upper bound: 4551.4009265
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -4552.2690723, upper bound: 4552.2690723
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -4552.2690723, upper bound: 4552.2690838
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -4552.1796248, upper bound: 4552.1801595
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -4552.1797752, upper bound: 4552.1796773
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 0, lower bound: -4551.2941722, upper bound: 4551.2889280
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 0, lower bound: -4551.2941722, upper bound: 4551.2891986

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5081802, upper bound: 4551.5022354
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5022354, upper bound: 4551.5022354
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2583493, upper bound: 4552.2582109
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2646966, upper bound: 4552.2598726
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2673117, upper bound: 4552.2673117
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2673117, upper bound: 4552.2673117
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1770530, upper bound: 4552.1770643
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1770530, upper bound: 4552.1770645
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1777914, upper bound: 4552.1781951
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1777914, upper bound: 4552.1778395
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1794311, upper bound: 4552.1791741
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1793438, upper bound: 4552.1791741
time: 0.75 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.58 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.58
Output dim: 0, lower bound: -4551.5081802, upper bound: 4551.5022354
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.58
Output dim: 0, lower bound: -4551.5022354, upper bound: 4551.5022354
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 0, lower bound: -4552.2583493, upper bound: 4552.2582109
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 0, lower bound: -4552.2646966, upper bound: 4552.2598726
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 0, lower bound: -4552.2673117, upper bound: 4552.2673117
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 0, lower bound: -4552.2673117, upper bound: 4552.2673117
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 0, lower bound: -4552.1770530, upper bound: 4552.1770643
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 0, lower bound: -4552.1770530, upper bound: 4552.1770645
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 0, lower bound: -4552.1777914, upper bound: 4552.1781951
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 0, lower bound: -4552.1777914, upper bound: 4552.1778395
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 0, lower bound: -4552.1794311, upper bound: 4552.1791741
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 0, lower bound: -4552.1793438, upper bound: 4552.1791741

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2583469, upper bound: 4552.2582085
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2582085, upper bound: 4552.2582085
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1766560, upper bound: 4552.1766560
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1823668, upper bound: 4552.1783781
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7908446, upper bound: 4551.7908446
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7908446, upper bound: 4551.7908446
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4549.5995026, upper bound: 4549.5995026
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4549.5995026, upper bound: 4549.5995026
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1770530, upper bound: 4552.1770530
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1770530, upper bound: 4552.1770643
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1770530, upper bound: 4552.1770530
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1770530, upper bound: 4552.1770645
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1774116, upper bound: 4552.1773778
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1773098, upper bound: 4552.1773778
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1776088, upper bound: 4552.1776527
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1776088, upper bound: 4552.1776088
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2320544, upper bound: 4551.2320544
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2320544, upper bound: 4551.2320544
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1791096, upper bound: 4552.1791096
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1793438, upper bound: 4552.1791741
time: 0.80 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.10 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 0, lower bound: -4552.2583469, upper bound: 4552.2582085
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 0, lower bound: -4552.2582085, upper bound: 4552.2582085
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 0, lower bound: -4552.1766560, upper bound: 4552.1766560
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 0, lower bound: -4552.1823668, upper bound: 4552.1783781
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 0, lower bound: -4551.7908446, upper bound: 4551.7908446
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 0, lower bound: -4551.7908446, upper bound: 4551.7908446
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 0, lower bound: -4549.5995026, upper bound: 4549.5995026
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 0, lower bound: -4549.5995026, upper bound: 4549.5995026
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 0, lower bound: -4552.1770530, upper bound: 4552.1770530
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 0, lower bound: -4552.1770530, upper bound: 4552.1770643
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 0, lower bound: -4552.1770530, upper bound: 4552.1770530
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 0, lower bound: -4552.1770530, upper bound: 4552.1770645
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 0, lower bound: -4552.1774116, upper bound: 4552.1773778
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 0, lower bound: -4552.1773098, upper bound: 4552.1773778
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 0, lower bound: -4552.1776088, upper bound: 4552.1776527
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 0, lower bound: -4552.1776088, upper bound: 4552.1776088
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 0, lower bound: -4551.2320544, upper bound: 4551.2320544
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.10
Output dim: 0, lower bound: -4551.2320544, upper bound: 4551.2320544
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 0, lower bound: -4552.1791096, upper bound: 4552.1791096
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.10
Output dim: 0, lower bound: -4552.1793438, upper bound: 4552.1791741

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4547.6013897, upper bound: 4547.6013897
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4547.6013897, upper bound: 4547.6013897
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4541.0859625, upper bound: 4541.0859625
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4541.0859625, upper bound: 4541.0859625
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4889960, upper bound: 4551.4889960
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4889960, upper bound: 4551.4889960
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1823610, upper bound: 4552.1766535
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1766535, upper bound: 4552.1783769
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6819673, upper bound: 4551.6819673
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6819673, upper bound: 4551.6819673
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.0278353, upper bound: 4551.0278353
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.0278353, upper bound: 4551.0278353
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4037775, upper bound: 4551.4037775
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4037775, upper bound: 4551.4037775
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2315695, upper bound: 4551.2315695
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2315695, upper bound: 4551.2315695
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1665379, upper bound: 4552.1665284
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1666354, upper bound: 4552.1665953
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1773098, upper bound: 4552.1773098
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1773098, upper bound: 4552.1773778
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1776088, upper bound: 4552.1776088
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1776088, upper bound: 4552.1776527
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4043567, upper bound: 4551.4043567
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4043567, upper bound: 4551.4043567
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0745649, upper bound: 4552.0745649
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0745649, upper bound: 4552.0745649
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1771072, upper bound: 4552.1768411
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1767743, upper bound: 4552.1768404
time: 0.86 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.79 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.79
Output dim: 0, lower bound: -4547.6013897, upper bound: 4547.6013897
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.79
Output dim: 0, lower bound: -4547.6013897, upper bound: 4547.6013897
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.79
Output dim: 0, lower bound: -4541.0859625, upper bound: 4541.0859625
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.79
Output dim: 0, lower bound: -4541.0859625, upper bound: 4541.0859625
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.79
Output dim: 0, lower bound: -4551.4889960, upper bound: 4551.4889960
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.79
Output dim: 0, lower bound: -4551.4889960, upper bound: 4551.4889960
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.79
Output dim: 0, lower bound: -4552.1823610, upper bound: 4552.1766535
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.79
Output dim: 0, lower bound: -4552.1766535, upper bound: 4552.1783769
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.79
Output dim: 0, lower bound: -4551.6819673, upper bound: 4551.6819673
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.79
Output dim: 0, lower bound: -4551.6819673, upper bound: 4551.6819673
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.79
Output dim: 0, lower bound: -4551.0278353, upper bound: 4551.0278353
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.79
Output dim: 0, lower bound: -4551.0278353, upper bound: 4551.0278353
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.79
Output dim: 0, lower bound: -4551.4037775, upper bound: 4551.4037775
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.79
Output dim: 0, lower bound: -4551.4037775, upper bound: 4551.4037775
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.79
Output dim: 0, lower bound: -4551.2315695, upper bound: 4551.2315695
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.79
Output dim: 0, lower bound: -4551.2315695, upper bound: 4551.2315695
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.79
Output dim: 0, lower bound: -4552.1665379, upper bound: 4552.1665284
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.79
Output dim: 0, lower bound: -4552.1666354, upper bound: 4552.1665953
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.79
Output dim: 0, lower bound: -4552.1773098, upper bound: 4552.1773098
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.79
Output dim: 0, lower bound: -4552.1773098, upper bound: 4552.1773778
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.79
Output dim: 0, lower bound: -4552.1776088, upper bound: 4552.1776088
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.79
Output dim: 0, lower bound: -4552.1776088, upper bound: 4552.1776527
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.79
Output dim: 0, lower bound: -4551.4043567, upper bound: 4551.4043567
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.79
Output dim: 0, lower bound: -4551.4043567, upper bound: 4551.4043567
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.79
Output dim: 0, lower bound: -4552.0745649, upper bound: 4552.0745649
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.79
Output dim: 0, lower bound: -4552.0745649, upper bound: 4552.0745649
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.79
Output dim: 0, lower bound: -4552.1771072, upper bound: 4552.1768411
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.79
Output dim: 0, lower bound: -4552.1767743, upper bound: 4552.1768404

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1766535, upper bound: 4552.1766535
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1822955, upper bound: 4552.1766535
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1766535, upper bound: 4552.1783769
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1766535, upper bound: 4552.1766535
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0623406, upper bound: 4552.0623231
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0623388, upper bound: 4552.0623231
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2223144, upper bound: 4551.2223144
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2223144, upper bound: 4551.2223144
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1765734, upper bound: 4552.1765734
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1765734, upper bound: 4552.1765734
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1765267, upper bound: 4552.1765943
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1765267, upper bound: 4552.1765267
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1765267, upper bound: 4552.1765267
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1765267, upper bound: 4552.1765267
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0716340, upper bound: 4552.0716784
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0716340, upper bound: 4552.0716340
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.0622371, upper bound: 4551.0622371
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.0622371, upper bound: 4551.0622371
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0745526, upper bound: 4552.0745526
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0745526, upper bound: 4552.0745526
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.9293676, upper bound: 4550.9293676
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.9293676, upper bound: 4550.9293676
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1767743, upper bound: 4552.1768399
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1767743, upper bound: 4552.1767743
time: 0.84 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 3.88 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.88
Output dim: 0, lower bound: -4552.1766535, upper bound: 4552.1766535
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.88
Output dim: 0, lower bound: -4552.1822955, upper bound: 4552.1766535
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.88
Output dim: 0, lower bound: -4552.1766535, upper bound: 4552.1783769
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.88
Output dim: 0, lower bound: -4552.1766535, upper bound: 4552.1766535
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.88
Output dim: 0, lower bound: -4552.0623406, upper bound: 4552.0623231
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.88
Output dim: 0, lower bound: -4552.0623388, upper bound: 4552.0623231
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.88
Output dim: 0, lower bound: -4551.2223144, upper bound: 4551.2223144
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.88
Output dim: 0, lower bound: -4551.2223144, upper bound: 4551.2223144
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.88
Output dim: 0, lower bound: -4552.1765734, upper bound: 4552.1765734
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.88
Output dim: 0, lower bound: -4552.1765734, upper bound: 4552.1765734
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.88
Output dim: 0, lower bound: -4552.1765267, upper bound: 4552.1765943
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.88
Output dim: 0, lower bound: -4552.1765267, upper bound: 4552.1765267
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.88
Output dim: 0, lower bound: -4552.1765267, upper bound: 4552.1765267
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.88
Output dim: 0, lower bound: -4552.1765267, upper bound: 4552.1765267
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.88
Output dim: 0, lower bound: -4552.0716340, upper bound: 4552.0716784
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.88
Output dim: 0, lower bound: -4552.0716340, upper bound: 4552.0716340
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.88
Output dim: 0, lower bound: -4551.0622371, upper bound: 4551.0622371
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.88
Output dim: 0, lower bound: -4551.0622371, upper bound: 4551.0622371
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.88
Output dim: 0, lower bound: -4552.0745526, upper bound: 4552.0745526
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.88
Output dim: 0, lower bound: -4552.0745526, upper bound: 4552.0745526
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.88
Output dim: 0, lower bound: -4550.9293676, upper bound: 4550.9293676
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.88
Output dim: 0, lower bound: -4550.9293676, upper bound: 4550.9293676
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.88
Output dim: 0, lower bound: -4552.1767743, upper bound: 4552.1768399
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.88
Output dim: 0, lower bound: -4552.1767743, upper bound: 4552.1767743

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1761332, upper bound: 4552.1761332
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1761332, upper bound: 4552.1761332
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1760758, upper bound: 4552.1760230
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1800781, upper bound: 4552.1760230
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4857288, upper bound: 4551.4857288
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4857288, upper bound: 4551.4857288
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1742579, upper bound: 4552.1742579
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1742579, upper bound: 4552.1742660
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0618902, upper bound: 4552.0616468
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0616468, upper bound: 4552.0616468
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0617273, upper bound: 4552.0616468
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0616468, upper bound: 4552.0616468
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1742603, upper bound: 4552.1742603
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1742603, upper bound: 4552.1742603
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2314079, upper bound: 4551.2314079
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2314079, upper bound: 4551.2314079
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6805103, upper bound: 4551.6805103
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6805103, upper bound: 4551.6805718
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1764919, upper bound: 4552.1764919
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1764919, upper bound: 4552.1764919
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.9235537, upper bound: 4550.9235537
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.9235537, upper bound: 4550.9235537
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.9235537, upper bound: 4550.9235537
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.9235537, upper bound: 4550.9235537
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0716099, upper bound: 4552.0716099
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0716099, upper bound: 4552.0716558
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.0620461, upper bound: 4551.0620461
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.0620461, upper bound: 4551.0620461
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.0570610, upper bound: 4551.0570610
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.0570610, upper bound: 4551.0570610
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0740111, upper bound: 4552.0740111
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0740111, upper bound: 4552.0740111
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6701585, upper bound: 4551.6701585
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6701585, upper bound: 4551.6702246
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0694134, upper bound: 4552.0694134
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0694134, upper bound: 4552.0694134
time: 0.88 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 4.17 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 4.17
Output dim: 0, lower bound: -4552.1761332, upper bound: 4552.1761332
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 4.17
Output dim: 0, lower bound: -4552.1761332, upper bound: 4552.1761332
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 4.17
Output dim: 0, lower bound: -4552.1760758, upper bound: 4552.1760230
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 4.17
Output dim: 0, lower bound: -4552.1800781, upper bound: 4552.1760230
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 4.17
Output dim: 0, lower bound: -4551.4857288, upper bound: 4551.4857288
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 4.17
Output dim: 0, lower bound: -4551.4857288, upper bound: 4551.4857288
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 4.17
Output dim: 0, lower bound: -4552.1742579, upper bound: 4552.1742579
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 4.17
Output dim: 0, lower bound: -4552.1742579, upper bound: 4552.1742660
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 4.17
Output dim: 0, lower bound: -4552.0618902, upper bound: 4552.0616468
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 4.17
Output dim: 0, lower bound: -4552.0616468, upper bound: 4552.0616468
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 4.17
Output dim: 0, lower bound: -4552.0617273, upper bound: 4552.0616468
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 4.17
Output dim: 0, lower bound: -4552.0616468, upper bound: 4552.0616468
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 4.17
Output dim: 0, lower bound: -4552.1742603, upper bound: 4552.1742603
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 4.17
Output dim: 0, lower bound: -4552.1742603, upper bound: 4552.1742603
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 4.17
Output dim: 0, lower bound: -4551.2314079, upper bound: 4551.2314079
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 4.17
Output dim: 0, lower bound: -4551.2314079, upper bound: 4551.2314079
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 4.17
Output dim: 0, lower bound: -4551.6805103, upper bound: 4551.6805103
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 4.17
Output dim: 0, lower bound: -4551.6805103, upper bound: 4551.6805718
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 4.17
Output dim: 0, lower bound: -4552.1764919, upper bound: 4552.1764919
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 4.17
Output dim: 0, lower bound: -4552.1764919, upper bound: 4552.1764919
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 4.17
Output dim: 0, lower bound: -4550.9235537, upper bound: 4550.9235537
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 4.17
Output dim: 0, lower bound: -4550.9235537, upper bound: 4550.9235537
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 4.17
Output dim: 0, lower bound: -4550.9235537, upper bound: 4550.9235537
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 4.17
Output dim: 0, lower bound: -4550.9235537, upper bound: 4550.9235537
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 4.17
Output dim: 0, lower bound: -4552.0716099, upper bound: 4552.0716099
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 4.17
Output dim: 0, lower bound: -4552.0716099, upper bound: 4552.0716558
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 4.17
Output dim: 0, lower bound: -4551.0620461, upper bound: 4551.0620461
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 4.17
Output dim: 0, lower bound: -4551.0620461, upper bound: 4551.0620461
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 4.17
Output dim: 0, lower bound: -4551.0570610, upper bound: 4551.0570610
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 4.17
Output dim: 0, lower bound: -4551.0570610, upper bound: 4551.0570610
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 4.17
Output dim: 0, lower bound: -4552.0740111, upper bound: 4552.0740111
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 4.17
Output dim: 0, lower bound: -4552.0740111, upper bound: 4552.0740111
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 4.17
Output dim: 0, lower bound: -4551.6701585, upper bound: 4551.6701585
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 4.17
Output dim: 0, lower bound: -4551.6701585, upper bound: 4551.6702246
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 4.17
Output dim: 0, lower bound: -4552.0694134, upper bound: 4552.0694134
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 4.17
Output dim: 0, lower bound: -4552.0694134, upper bound: 4552.0694134

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7006979, upper bound: 4551.7006979
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7006979, upper bound: 4551.7006979
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1655075, upper bound: 4552.1655075
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1655075, upper bound: 4552.1655075
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1755027, upper bound: 4552.1755027
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1755042, upper bound: 4552.1755027
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1650971, upper bound: 4552.1650971
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1691521, upper bound: 4552.1650971
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1634524, upper bound: 4552.1634524
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1634524, upper bound: 4552.1634524
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7012041, upper bound: 4551.7012041
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7012041, upper bound: 4551.7012123
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0598388, upper bound: 4552.0595876
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0595876, upper bound: 4552.0595876
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0590838, upper bound: 4552.0590838
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0590838, upper bound: 4552.0590838
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3041971, upper bound: 4551.3041876
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3041969, upper bound: 4551.3041876
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0616122, upper bound: 4552.0616122
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0616122, upper bound: 4552.0616122
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4017499, upper bound: 4551.4017499
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4017499, upper bound: 4551.4017499
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6783218, upper bound: 4551.6783218
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6783218, upper bound: 4551.6783218
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3921735, upper bound: 4551.3921735
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3921735, upper bound: 4551.3921735
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1744804, upper bound: 4552.1744804
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1744804, upper bound: 4552.1744804
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.7933711, upper bound: 4550.7933711
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.7933711, upper bound: 4550.7933711
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3001736, upper bound: 4551.3001736
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3001736, upper bound: 4551.3001736
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0707817, upper bound: 4552.0707817
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0707817, upper bound: 4552.0707817
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0712929, upper bound: 4552.0712929
time: 3.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0712929, upper bound: 4552.0712929
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0599854, upper bound: 4552.0599854
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0599854, upper bound: 4552.0599854
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0690583, upper bound: 4552.0690583
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0690583, upper bound: 4552.0690583
time: 0.85 seconds

## Summary of splitting (split count: 11)
- Time for RS candidates: 4.27 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.27
Output dim: 0, lower bound: -4551.7006979, upper bound: 4551.7006979
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.27
Output dim: 0, lower bound: -4551.7006979, upper bound: 4551.7006979
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.1655075, upper bound: 4552.1655075
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.1655075, upper bound: 4552.1655075
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.1755027, upper bound: 4552.1755027
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.1755042, upper bound: 4552.1755027
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.1650971, upper bound: 4552.1650971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.1691521, upper bound: 4552.1650971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.1634524, upper bound: 4552.1634524
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.1634524, upper bound: 4552.1634524
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.27
Output dim: 0, lower bound: -4551.7012041, upper bound: 4551.7012041
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.27
Output dim: 0, lower bound: -4551.7012041, upper bound: 4551.7012123
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.0598388, upper bound: 4552.0595876
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.0595876, upper bound: 4552.0595876
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.0590838, upper bound: 4552.0590838
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.0590838, upper bound: 4552.0590838
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.27
Output dim: 0, lower bound: -4551.3041971, upper bound: 4551.3041876
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.27
Output dim: 0, lower bound: -4551.3041969, upper bound: 4551.3041876
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.0616122, upper bound: 4552.0616122
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.0616122, upper bound: 4552.0616122
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.27
Output dim: 0, lower bound: -4551.4017499, upper bound: 4551.4017499
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.27
Output dim: 0, lower bound: -4551.4017499, upper bound: 4551.4017499
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.27
Output dim: 0, lower bound: -4551.6783218, upper bound: 4551.6783218
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.27
Output dim: 0, lower bound: -4551.6783218, upper bound: 4551.6783218
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.27
Output dim: 0, lower bound: -4551.3921735, upper bound: 4551.3921735
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.27
Output dim: 0, lower bound: -4551.3921735, upper bound: 4551.3921735
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.1744804, upper bound: 4552.1744804
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.1744804, upper bound: 4552.1744804
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.27
Output dim: 0, lower bound: -4550.7933711, upper bound: 4550.7933711
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.27
Output dim: 0, lower bound: -4550.7933711, upper bound: 4550.7933711
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.27
Output dim: 0, lower bound: -4551.3001736, upper bound: 4551.3001736
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.27
Output dim: 0, lower bound: -4551.3001736, upper bound: 4551.3001736
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.0707817, upper bound: 4552.0707817
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.0707817, upper bound: 4552.0707817
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.0712929, upper bound: 4552.0712929
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.0712929, upper bound: 4552.0712929
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.0599854, upper bound: 4552.0599854
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.0599854, upper bound: 4552.0599854
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.0690583, upper bound: 4552.0690583
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 12, time: 4.27
Output dim: 0, lower bound: -4552.0690583, upper bound: 4552.0690583

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0663471, upper bound: 4552.0663471
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0663471, upper bound: 4552.0663471
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4794290, upper bound: 4551.4794290
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4794290, upper bound: 4551.4794290
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1755027, upper bound: 4552.1755027
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1755027, upper bound: 4552.1755027
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1755027, upper bound: 4552.1755027
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1755042, upper bound: 4552.1755027
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0594170, upper bound: 4552.0594170
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0594170, upper bound: 4552.0594170
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6663290, upper bound: 4551.6662331
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6673097, upper bound: 4551.6662331
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 4): status=Status.UNKNOWN, low=0.9375000, high=0.9687500, mid=0.9687500, abs_max=5687.5751953125
rel_dist={0: [-4552.289001211694, 4552.289001211693]}

## Binary search (step 5) starts
Candidate diff: 0.9531250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2883619, upper bound: 4552.2883619
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2883619, upper bound: 4552.2883619
time: 0.72 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.46 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 0, lower bound: -4552.2883619, upper bound: 4552.2883619
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 0, lower bound: -4552.2883619, upper bound: 4552.2883619

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6132877, upper bound: 4551.6132877
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6132877, upper bound: 4551.6132877
time: 0.75 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2883619, upper bound: 4552.2802851
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2802851, upper bound: 4552.2883619
time: 0.79 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.77 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.77
Output dim: 0, lower bound: -4551.6132877, upper bound: 4551.6132877
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.77
Output dim: 0, lower bound: -4551.6132877, upper bound: 4551.6132877
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.77
Output dim: 0, lower bound: -4552.2883619, upper bound: 4552.2802851
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.77
Output dim: 0, lower bound: -4552.2802851, upper bound: 4552.2883619

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2789108, upper bound: 4552.2798120
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2824628, upper bound: 4552.2796790
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2690538, upper bound: 4552.2764711
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2690394, upper bound: 4552.2764711
time: 0.69 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.51 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 0, lower bound: -4552.2789108, upper bound: 4552.2798120
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 0, lower bound: -4552.2824628, upper bound: 4552.2796790
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 0, lower bound: -4552.2690538, upper bound: 4552.2764711
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 0, lower bound: -4552.2690394, upper bound: 4552.2764711

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4548.8495689, upper bound: 4548.8495689
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4548.8495689, upper bound: 4548.8495689
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2812639, upper bound: 4552.2782571
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2793748, upper bound: 4552.2766516
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2608516, upper bound: 4552.2750947
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2666598, upper bound: 4552.2608754
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1873414, upper bound: 4552.1905252
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1872807, upper bound: 4552.1902392
time: 0.81 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.99 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.99
Output dim: 0, lower bound: -4548.8495689, upper bound: 4548.8495689
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.99
Output dim: 0, lower bound: -4548.8495689, upper bound: 4548.8495689
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 0, lower bound: -4552.2812639, upper bound: 4552.2782571
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 0, lower bound: -4552.2793748, upper bound: 4552.2766516
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 0, lower bound: -4552.2608516, upper bound: 4552.2750947
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 0, lower bound: -4552.2666598, upper bound: 4552.2608754
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 0, lower bound: -4552.1873414, upper bound: 4552.1905252
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.99
Output dim: 0, lower bound: -4552.1872807, upper bound: 4552.1902392

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5505807, upper bound: 4551.5503017
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5497222, upper bound: 4551.5515464
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2590664, upper bound: 4552.2620545
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2628677, upper bound: 4552.2590768
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2608062, upper bound: 4552.2607778
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2607778, upper bound: 4552.2750592
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5087799, upper bound: 4551.5022363
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5024634, upper bound: 4551.5022363
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1873414, upper bound: 4552.1803854
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1792103, upper bound: 4552.1905219
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.0728712, upper bound: 4551.0771800
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.0704562, upper bound: 4551.0771800
time: 0.91 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.74 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.74
Output dim: 0, lower bound: -4551.5505807, upper bound: 4551.5503017
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.74
Output dim: 0, lower bound: -4551.5497222, upper bound: 4551.5515464
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 0, lower bound: -4552.2590664, upper bound: 4552.2620545
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 0, lower bound: -4552.2628677, upper bound: 4552.2590768
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 0, lower bound: -4552.2608062, upper bound: 4552.2607778
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 0, lower bound: -4552.2607778, upper bound: 4552.2750592
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.74
Output dim: 0, lower bound: -4551.5087799, upper bound: 4551.5022363
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.74
Output dim: 0, lower bound: -4551.5024634, upper bound: 4551.5022363
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 0, lower bound: -4552.1873414, upper bound: 4552.1803854
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 0, lower bound: -4552.1792103, upper bound: 4552.1905219
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.74
Output dim: 0, lower bound: -4551.0728712, upper bound: 4551.0771800
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.74
Output dim: 0, lower bound: -4551.0704562, upper bound: 4551.0771800

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7664077, upper bound: 4551.7665397
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7664077, upper bound: 4551.7702648
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4548.8483823, upper bound: 4548.8483823
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4548.8483823, upper bound: 4548.8483823
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.9261635, upper bound: 4550.9261635
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.9261635, upper bound: 4550.9261635
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4546.3302333, upper bound: 4546.3302333
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4546.3302333, upper bound: 4546.3302333
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1741744, upper bound: 4552.1697597
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1767157, upper bound: 4552.1685846
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1792103, upper bound: 4552.1887405
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1792103, upper bound: 4552.1792625
time: 0.85 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.86 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -4551.7664077, upper bound: 4551.7665397
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -4551.7664077, upper bound: 4551.7702648
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -4548.8483823, upper bound: 4548.8483823
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -4548.8483823, upper bound: 4548.8483823
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -4550.9261635, upper bound: 4550.9261635
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -4550.9261635, upper bound: 4550.9261635
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -4546.3302333, upper bound: 4546.3302333
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -4546.3302333, upper bound: 4546.3302333
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -4552.1741744, upper bound: 4552.1697597
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -4552.1767157, upper bound: 4552.1685846
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -4552.1792103, upper bound: 4552.1887405
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -4552.1792103, upper bound: 4552.1792625

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4824887, upper bound: 4551.4826543
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4824764, upper bound: 4551.4826543
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1685846, upper bound: 4552.1685846
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1744667, upper bound: 4552.1685846
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1785837, upper bound: 4552.1873631
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1785837, upper bound: 4552.1841643
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1785837, upper bound: 4552.1785837
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1785837, upper bound: 4552.1786360
time: 0.86 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.52 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.52
Output dim: 0, lower bound: -4551.4824887, upper bound: 4551.4826543
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.52
Output dim: 0, lower bound: -4551.4824764, upper bound: 4551.4826543
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.52
Output dim: 0, lower bound: -4552.1685846, upper bound: 4552.1685846
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.52
Output dim: 0, lower bound: -4552.1744667, upper bound: 4552.1685846
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.52
Output dim: 0, lower bound: -4552.1785837, upper bound: 4552.1873631
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.52
Output dim: 0, lower bound: -4552.1785837, upper bound: 4552.1841643
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.52
Output dim: 0, lower bound: -4552.1785837, upper bound: 4552.1785837
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.52
Output dim: 0, lower bound: -4552.1785837, upper bound: 4552.1786360

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4824065, upper bound: 4551.4824065
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4824065, upper bound: 4551.4824065
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4892898, upper bound: 4551.4824065
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4859550, upper bound: 4551.4824065
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0713860, upper bound: 4552.0754421
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0713860, upper bound: 4552.0800440
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1785837, upper bound: 4552.1841643
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1785837, upper bound: 4552.1790371
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1782806, upper bound: 4552.1782806
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1782806, upper bound: 4552.1782806
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1768099, upper bound: 4552.1768622
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1768099, upper bound: 4552.1768503
time: 0.85 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.59 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.59
Output dim: 0, lower bound: -4551.4824065, upper bound: 4551.4824065
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.59
Output dim: 0, lower bound: -4551.4824065, upper bound: 4551.4824065
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.59
Output dim: 0, lower bound: -4551.4892898, upper bound: 4551.4824065
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.59
Output dim: 0, lower bound: -4551.4859550, upper bound: 4551.4824065
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.59
Output dim: 0, lower bound: -4552.0713860, upper bound: 4552.0754421
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.59
Output dim: 0, lower bound: -4552.0713860, upper bound: 4552.0800440
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.59
Output dim: 0, lower bound: -4552.1785837, upper bound: 4552.1841643
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.59
Output dim: 0, lower bound: -4552.1785837, upper bound: 4552.1790371
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.59
Output dim: 0, lower bound: -4552.1782806, upper bound: 4552.1782806
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.59
Output dim: 0, lower bound: -4552.1782806, upper bound: 4552.1782806
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.59
Output dim: 0, lower bound: -4552.1768099, upper bound: 4552.1768622
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.59
Output dim: 0, lower bound: -4552.1768099, upper bound: 4552.1768503

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0708975, upper bound: 4552.0708975
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0708975, upper bound: 4552.0749535
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6001277, upper bound: 4551.6071057
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6001277, upper bound: 4551.6084608
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.0575331, upper bound: 4551.0580720
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.0575331, upper bound: 4551.0580720
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2589741, upper bound: 4551.2589843
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2589741, upper bound: 4551.2589843
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1782806, upper bound: 4552.1782806
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1782806, upper bound: 4552.1782806
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6792737, upper bound: 4551.6792737
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6792737, upper bound: 4551.6792737
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1764919, upper bound: 4552.1765354
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1764919, upper bound: 4552.1764919
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2589741, upper bound: 4551.2589741
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2589741, upper bound: 4551.2589741
time: 0.88 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.52 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.52
Output dim: 0, lower bound: -4552.0708975, upper bound: 4552.0708975
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.52
Output dim: 0, lower bound: -4552.0708975, upper bound: 4552.0749535
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.52
Output dim: 0, lower bound: -4551.6001277, upper bound: 4551.6071057
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.52
Output dim: 0, lower bound: -4551.6001277, upper bound: 4551.6084608
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.52
Output dim: 0, lower bound: -4551.0575331, upper bound: 4551.0580720
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.52
Output dim: 0, lower bound: -4551.0575331, upper bound: 4551.0580720
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.52
Output dim: 0, lower bound: -4551.2589741, upper bound: 4551.2589843
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.52
Output dim: 0, lower bound: -4551.2589741, upper bound: 4551.2589843
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.52
Output dim: 0, lower bound: -4552.1782806, upper bound: 4552.1782806
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.52
Output dim: 0, lower bound: -4552.1782806, upper bound: 4552.1782806
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.52
Output dim: 0, lower bound: -4551.6792737, upper bound: 4551.6792737
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.52
Output dim: 0, lower bound: -4551.6792737, upper bound: 4551.6792737
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.52
Output dim: 0, lower bound: -4552.1764919, upper bound: 4552.1765354
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.52
Output dim: 0, lower bound: -4552.1764919, upper bound: 4552.1764919
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.52
Output dim: 0, lower bound: -4551.2589741, upper bound: 4551.2589741
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.52
Output dim: 0, lower bound: -4551.2589741, upper bound: 4551.2589741

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3748066, upper bound: 4551.3748066
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3748066, upper bound: 4551.3748066
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1276655, upper bound: 4551.1277860
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1276655, upper bound: 4551.1315916
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.9075567, upper bound: 4550.9075567
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.9075567, upper bound: 4550.9075567
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1764919, upper bound: 4552.1764919
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1764919, upper bound: 4552.1764919
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1656995, upper bound: 4552.1657430
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1656995, upper bound: 4552.1657178
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1764919, upper bound: 4552.1764919
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1764919, upper bound: 4552.1764919
time: 0.99 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 4.92 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 4.92
Output dim: 0, lower bound: -4551.3748066, upper bound: 4551.3748066
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 4.92
Output dim: 0, lower bound: -4551.3748066, upper bound: 4551.3748066
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 4.92
Output dim: 0, lower bound: -4551.1276655, upper bound: 4551.1277860
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 4.92
Output dim: 0, lower bound: -4551.1276655, upper bound: 4551.1315916
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 4.92
Output dim: 0, lower bound: -4550.9075567, upper bound: 4550.9075567
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 4.92
Output dim: 0, lower bound: -4550.9075567, upper bound: 4550.9075567
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 4.92
Output dim: 0, lower bound: -4552.1764919, upper bound: 4552.1764919
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 4.92
Output dim: 0, lower bound: -4552.1764919, upper bound: 4552.1764919
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 4.92
Output dim: 0, lower bound: -4552.1656995, upper bound: 4552.1657430
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 4.92
Output dim: 0, lower bound: -4552.1656995, upper bound: 4552.1657178
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 4.92
Output dim: 0, lower bound: -4552.1764919, upper bound: 4552.1764919
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 4.92
Output dim: 0, lower bound: -4552.1764919, upper bound: 4552.1764919

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1744804, upper bound: 4552.1744804
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1744804, upper bound: 4552.1744804
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3883738, upper bound: 4551.3883738
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3883738, upper bound: 4551.3883738
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4721936, upper bound: 4551.4721936
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4721936, upper bound: 4551.4721936
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1656995, upper bound: 4552.1656995
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1656995, upper bound: 4552.1657178
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4785780, upper bound: 4551.4785780
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4785780, upper bound: 4551.4785780
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1764919, upper bound: 4552.1764919
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1764919, upper bound: 4552.1764919
time: 0.77 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 3.51 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.51
Output dim: 0, lower bound: -4552.1744804, upper bound: 4552.1744804
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.51
Output dim: 0, lower bound: -4552.1744804, upper bound: 4552.1744804
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.51
Output dim: 0, lower bound: -4551.3883738, upper bound: 4551.3883738
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.51
Output dim: 0, lower bound: -4551.3883738, upper bound: 4551.3883738
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.51
Output dim: 0, lower bound: -4551.4721936, upper bound: 4551.4721936
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.51
Output dim: 0, lower bound: -4551.4721936, upper bound: 4551.4721936
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.51
Output dim: 0, lower bound: -4552.1656995, upper bound: 4552.1656995
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.51
Output dim: 0, lower bound: -4552.1656995, upper bound: 4552.1657178
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.51
Output dim: 0, lower bound: -4551.4785780, upper bound: 4551.4785780
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.51
Output dim: 0, lower bound: -4551.4785780, upper bound: 4551.4785780
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.51
Output dim: 0, lower bound: -4552.1764919, upper bound: 4552.1764919
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.51
Output dim: 0, lower bound: -4552.1764919, upper bound: 4552.1764919

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2238109, upper bound: 4551.2238109
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2238109, upper bound: 4551.2238109
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6696437, upper bound: 4551.6696437
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6696437, upper bound: 4551.6696437
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1636688, upper bound: 4552.1636688
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1636688, upper bound: 4552.1636688
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1649628, upper bound: 4552.1649628
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1649628, upper bound: 4552.1649810
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.8951214, upper bound: 4550.8951214
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.8951214, upper bound: 4550.8951214
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3883738, upper bound: 4551.3883738
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3883738, upper bound: 4551.3883738
time: 0.95 seconds

## Summary of splitting (split count: 11)
- Time for RS candidates: 3.73 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 3.73
Output dim: 0, lower bound: -4551.2238109, upper bound: 4551.2238109
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 3.73
Output dim: 0, lower bound: -4551.2238109, upper bound: 4551.2238109
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 3.73
Output dim: 0, lower bound: -4551.6696437, upper bound: 4551.6696437
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 3.73
Output dim: 0, lower bound: -4551.6696437, upper bound: 4551.6696437
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 12, time: 3.73
Output dim: 0, lower bound: -4552.1636688, upper bound: 4552.1636688
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 12, time: 3.73
Output dim: 0, lower bound: -4552.1636688, upper bound: 4552.1636688
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 12, time: 3.73
Output dim: 0, lower bound: -4552.1649628, upper bound: 4552.1649628
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 12, time: 3.73
Output dim: 0, lower bound: -4552.1649628, upper bound: 4552.1649810
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 3.73
Output dim: 0, lower bound: -4550.8951214, upper bound: 4550.8951214
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 3.73
Output dim: 0, lower bound: -4550.8951214, upper bound: 4550.8951214
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 3.73
Output dim: 0, lower bound: -4551.3883738, upper bound: 4551.3883738
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 3.73
Output dim: 0, lower bound: -4551.3883738, upper bound: 4551.3883738

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1636688, upper bound: 4552.1636688
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1636688, upper bound: 4552.1636688
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1636688, upper bound: 4552.1636688
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1636688, upper bound: 4552.1636688
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2153152, upper bound: 4551.2153152
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2153152, upper bound: 4551.2153152
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6603724, upper bound: 4551.6603724
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6603724, upper bound: 4551.6603881
time: 0.75 seconds

## Summary of splitting (split count: 12)
- Time for RS candidates: 3.08 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 13, time: 3.08
Output dim: 0, lower bound: -4552.1636688, upper bound: 4552.1636688
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 13, time: 3.08
Output dim: 0, lower bound: -4552.1636688, upper bound: 4552.1636688
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 13, time: 3.08
Output dim: 0, lower bound: -4552.1636688, upper bound: 4552.1636688
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 13, time: 3.08
Output dim: 0, lower bound: -4552.1636688, upper bound: 4552.1636688
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 13, time: 3.08
Output dim: 0, lower bound: -4551.2153152, upper bound: 4551.2153152
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 13, time: 3.08
Output dim: 0, lower bound: -4551.2153152, upper bound: 4551.2153152
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 13, time: 3.08
Output dim: 0, lower bound: -4551.6603724, upper bound: 4551.6603724
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 13, time: 3.08
Output dim: 0, lower bound: -4551.6603724, upper bound: 4551.6603881

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0570113, upper bound: 4552.0570113
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0570113, upper bound: 4552.0570113
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2140710, upper bound: 4551.2140710
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2140710, upper bound: 4551.2140710
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1636688, upper bound: 4552.1636688
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1636688, upper bound: 4552.1636688
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2140710, upper bound: 4551.2140710
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2140710, upper bound: 4551.2140710
time: 0.75 seconds

## Summary of splitting (split count: 13)
- Time for RS candidates: 3.12 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 14, time: 3.12
Output dim: 0, lower bound: -4552.0570113, upper bound: 4552.0570113
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 14, time: 3.12
Output dim: 0, lower bound: -4552.0570113, upper bound: 4552.0570113
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 14, time: 3.12
Output dim: 0, lower bound: -4551.2140710, upper bound: 4551.2140710
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 14, time: 3.12
Output dim: 0, lower bound: -4551.2140710, upper bound: 4551.2140710
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 14, time: 3.12
Output dim: 0, lower bound: -4552.1636688, upper bound: 4552.1636688
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 14, time: 3.12
Output dim: 0, lower bound: -4552.1636688, upper bound: 4552.1636688
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 14, time: 3.12
Output dim: 0, lower bound: -4551.2140710, upper bound: 4551.2140710
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 14, time: 3.12
Output dim: 0, lower bound: -4551.2140710, upper bound: 4551.2140710

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.7578973, upper bound: 4550.7578973
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.7578973, upper bound: 4550.7578973
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.0458961, upper bound: 4551.0458961
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.0458961, upper bound: 4551.0458961
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2140710, upper bound: 4551.2140710
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2140710, upper bound: 4551.2140710
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3805794, upper bound: 4551.3805794
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3805794, upper bound: 4551.3805794
time: 0.80 seconds

## Summary of splitting (split count: 14)
- Time for RS candidates: 3.29 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 15, time: 3.29
Output dim: 0, lower bound: -4550.7578973, upper bound: 4550.7578973
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 15, time: 3.29
Output dim: 0, lower bound: -4550.7578973, upper bound: 4550.7578973
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 15, time: 3.29
Output dim: 0, lower bound: -4551.0458961, upper bound: 4551.0458961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 15, time: 3.29
Output dim: 0, lower bound: -4551.0458961, upper bound: 4551.0458961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 15, time: 3.29
Output dim: 0, lower bound: -4551.2140710, upper bound: 4551.2140710
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 15, time: 3.29
Output dim: 0, lower bound: -4551.2140710, upper bound: 4551.2140710
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 15, time: 3.29
Output dim: 0, lower bound: -4551.3805794, upper bound: 4551.3805794
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 15, time: 3.29
Output dim: 0, lower bound: -4551.3805794, upper bound: 4551.3805794
Binary search (step 5): status=Status.VERIFIED, low=0.9531250, high=0.9687500, mid=0.9531250, abs_max=5687.5751953125
rel_dist={0: [-4552.289001211694, 4552.289001211693]}

## Binary search (step 6) starts
Candidate diff: 0.9609375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1776004, upper bound: 4551.1775821
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1775821, upper bound: 4551.1776004
time: 0.77 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.63 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.63
Output dim: 0, lower bound: -4551.1776004, upper bound: 4551.1775821
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.63
Output dim: 0, lower bound: -4551.1775821, upper bound: 4551.1776004
Binary search (step 6): status=Status.VERIFIED, low=0.9609375, high=0.9687500, mid=0.9609375, abs_max=5687.5751953125
rel_dist={0: [-4552.289001211694, 4552.289001211693]}

## Binary search (step 7) starts
Candidate diff: 0.9648438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2881480, upper bound: 4552.2884988
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2884988, upper bound: 4552.2881480
time: 0.78 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 0, lower bound: -4552.2881480, upper bound: 4552.2884988
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 0, lower bound: -4552.2884988, upper bound: 4552.2881480

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2734201, upper bound: 4552.2761444
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2757540, upper bound: 4552.2738354
time: 0.74 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4548.8675040, upper bound: 4548.8675040
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4548.8675040, upper bound: 4548.8675040
time: 0.83 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.11 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -4552.2734201, upper bound: 4552.2761444
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -4552.2757540, upper bound: 4552.2738354
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.11
Output dim: 0, lower bound: -4548.8675040, upper bound: 4548.8675040
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.11
Output dim: 0, lower bound: -4548.8675040, upper bound: 4548.8675040

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2734201, upper bound: 4552.2646270
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2692239, upper bound: 4552.2761444
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8356124, upper bound: 4551.8229940
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8311560, upper bound: 4551.8340809
time: 0.79 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.22 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -4552.2734201, upper bound: 4552.2646270
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -4552.2692239, upper bound: 4552.2761444
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -4551.8356124, upper bound: 4551.8229940
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -4551.8311560, upper bound: 4551.8340809

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1803656, upper bound: 4552.1839475
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1805169, upper bound: 4552.1839331
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2570813, upper bound: 4552.2676408
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2635375, upper bound: 4552.2615099
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.0953046, upper bound: 4551.0903829
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.0904600, upper bound: 4551.0903829
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4548.2660749, upper bound: 4548.2660749
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4548.2660749, upper bound: 4548.2660749
time: 1.04 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.56 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 0, lower bound: -4552.1803656, upper bound: 4552.1839475
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 0, lower bound: -4552.1805169, upper bound: 4552.1839331
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 0, lower bound: -4552.2570813, upper bound: 4552.2676408
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 0, lower bound: -4552.2635375, upper bound: 4552.2615099
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 0, lower bound: -4551.0953046, upper bound: 4551.0903829
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 0, lower bound: -4551.0904600, upper bound: 4551.0903829
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 0, lower bound: -4548.2660749, upper bound: 4548.2660749
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 0, lower bound: -4548.2660749, upper bound: 4548.2660749

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1785227, upper bound: 4552.1821160
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1785146, upper bound: 4552.1785493
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1803705, upper bound: 4552.1839331
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1805169, upper bound: 4552.1838872
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2570616, upper bound: 4552.2570482
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2570482, upper bound: 4552.2676151
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9347771, upper bound: 4551.9368654
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9315213, upper bound: 4551.9368654
time: 0.87 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.47 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 0, lower bound: -4552.1785227, upper bound: 4552.1821160
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 0, lower bound: -4552.1785146, upper bound: 4552.1785493
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 0, lower bound: -4552.1803705, upper bound: 4552.1839331
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 0, lower bound: -4552.1805169, upper bound: 4552.1838872
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 0, lower bound: -4552.2570616, upper bound: 4552.2570482
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 0, lower bound: -4552.2570482, upper bound: 4552.2676151
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 0, lower bound: -4551.9347771, upper bound: 4551.9368654
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 0, lower bound: -4551.9315213, upper bound: 4551.9368654

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1785227, upper bound: 4552.1821160
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1785146, upper bound: 4552.1785776
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1785040, upper bound: 4552.1785040
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1785040, upper bound: 4552.1785395
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1798784, upper bound: 4552.1833783
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1798837, upper bound: 4552.1833737
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1805169, upper bound: 4552.1838872
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1803557, upper bound: 4552.1804210
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2457280, upper bound: 4552.2457104
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2457280, upper bound: 4552.2457104
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1833495, upper bound: 4552.1929661
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1833495, upper bound: 4552.1934899
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4865129, upper bound: 4551.4865511
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4865129, upper bound: 4551.4927098
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9313529, upper bound: 4551.9367059
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9315213, upper bound: 4551.9357086
time: 0.81 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -4552.1785227, upper bound: 4552.1821160
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -4552.1785146, upper bound: 4552.1785776
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -4552.1785040, upper bound: 4552.1785040
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -4552.1785040, upper bound: 4552.1785395
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -4552.1798784, upper bound: 4552.1833783
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -4552.1798837, upper bound: 4552.1833737
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -4552.1805169, upper bound: 4552.1838872
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -4552.1803557, upper bound: 4552.1804210
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -4552.2457280, upper bound: 4552.2457104
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -4552.2457280, upper bound: 4552.2457104
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -4552.1833495, upper bound: 4552.1929661
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -4552.1833495, upper bound: 4552.1934899
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.32
Output dim: 0, lower bound: -4551.4865129, upper bound: 4551.4865511
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.32
Output dim: 0, lower bound: -4551.4865129, upper bound: 4551.4927098
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -4551.9313529, upper bound: 4551.9367059
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -4551.9315213, upper bound: 4551.9357086

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1785227, upper bound: 4552.1821160
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1785146, upper bound: 4552.1788979
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1677721, upper bound: 4552.1678414
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1677721, upper bound: 4552.1678396
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1785040, upper bound: 4552.1785040
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1785040, upper bound: 4552.1785040
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0725418, upper bound: 4552.0725778
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0725418, upper bound: 4552.0725418
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4884949, upper bound: 4551.4937876
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4884949, upper bound: 4551.4944841
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1786575, upper bound: 4552.1786293
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1786575, upper bound: 4552.1823902
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1800116, upper bound: 4552.1828868
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1796286, upper bound: 4552.1828523
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1803456, upper bound: 4552.1803436
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1803436, upper bound: 4552.1804142
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7557832, upper bound: 4551.7557832
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7557832, upper bound: 4551.7557832
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2450907, upper bound: 4552.2450732
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2450814, upper bound: 4552.2450732
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1827955, upper bound: 4552.1827955
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1827955, upper bound: 4552.1924261
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1833495, upper bound: 4552.1903507
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1833495, upper bound: 4552.1868039
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8617804, upper bound: 4551.8676208
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8617804, upper bound: 4551.8675268
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9308400, upper bound: 4551.9348578
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9306716, upper bound: 4551.9350227
time: 0.75 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4552.1785227, upper bound: 4552.1821160
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4552.1785146, upper bound: 4552.1788979
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4552.1677721, upper bound: 4552.1678414
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4552.1677721, upper bound: 4552.1678396
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4552.1785040, upper bound: 4552.1785040
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4552.1785040, upper bound: 4552.1785040
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4552.0725418, upper bound: 4552.0725778
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4552.0725418, upper bound: 4552.0725418
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 0, lower bound: -4551.4884949, upper bound: 4551.4937876
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 0, lower bound: -4551.4884949, upper bound: 4551.4944841
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4552.1786575, upper bound: 4552.1786293
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4552.1786575, upper bound: 4552.1823902
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4552.1800116, upper bound: 4552.1828868
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4552.1796286, upper bound: 4552.1828523
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4552.1803456, upper bound: 4552.1803436
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4552.1803436, upper bound: 4552.1804142
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 0, lower bound: -4551.7557832, upper bound: 4551.7557832
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.12
Output dim: 0, lower bound: -4551.7557832, upper bound: 4551.7557832
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4552.2450907, upper bound: 4552.2450732
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4552.2450814, upper bound: 4552.2450732
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4552.1827955, upper bound: 4552.1827955
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4552.1827955, upper bound: 4552.1924261
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4552.1833495, upper bound: 4552.1903507
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4552.1833495, upper bound: 4552.1868039
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4551.8617804, upper bound: 4551.8676208
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4551.8617804, upper bound: 4551.8675268
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4551.9308400, upper bound: 4551.9348578
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -4551.9306716, upper bound: 4551.9350227

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4190333, upper bound: 4551.4249917
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4190333, upper bound: 4551.4257827
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2508735, upper bound: 4551.2508735
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2508735, upper bound: 4551.2508735
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1659558, upper bound: 4552.1660277
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1659558, upper bound: 4552.1659558
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.9828090, upper bound: 4550.9828406
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.9828090, upper bound: 4550.9828625
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1779534, upper bound: 4552.1779534
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1779534, upper bound: 4552.1779534
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7177194, upper bound: 4551.7177194
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7177194, upper bound: 4551.7177194
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1290439, upper bound: 4551.1290439
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1290439, upper bound: 4551.1290439
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.9127582, upper bound: 4550.9127784
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.9127582, upper bound: 4550.9127582
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1786120, upper bound: 4552.1785837
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1785837, upper bound: 4552.1785837
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2656369, upper bound: 4551.2734672
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2656369, upper bound: 4551.2734672
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1791231, upper bound: 4552.1791096
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1791231, upper bound: 4552.1827321
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1777914, upper bound: 4552.1810945
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1777952, upper bound: 4552.1810199
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1785432, upper bound: 4552.1785288
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1785445, upper bound: 4552.1785288
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0758610, upper bound: 4552.0759294
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0758610, upper bound: 4552.0759286
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2444294, upper bound: 4552.2444123
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2444300, upper bound: 4552.2444123
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7548688, upper bound: 4551.7548688
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7548688, upper bound: 4551.7548688
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0720295, upper bound: 4552.0720295
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0720295, upper bound: 4552.0720295
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2055892, upper bound: 4551.2117781
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.2055892, upper bound: 4551.2117781
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1830724, upper bound: 4552.1893609
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1830724, upper bound: 4552.1893612
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9019164, upper bound: 4551.9019785
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9019164, upper bound: 4551.9053101
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8610584, upper bound: 4551.8656602
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8610584, upper bound: 4551.8668863
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7271371, upper bound: 4551.7271371
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7271371, upper bound: 4551.7322659
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8612279, upper bound: 4551.8655644
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.8610584, upper bound: 4551.8610584
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9287906, upper bound: 4551.9331253
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9287906, upper bound: 4551.9331252
time: 0.87 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.4190333, upper bound: 4551.4249917
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.4190333, upper bound: 4551.4257827
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.2508735, upper bound: 4551.2508735
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.2508735, upper bound: 4551.2508735
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4552.1659558, upper bound: 4552.1660277
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4552.1659558, upper bound: 4552.1659558
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.52
Output dim: 0, lower bound: -4550.9828090, upper bound: 4550.9828406
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.52
Output dim: 0, lower bound: -4550.9828090, upper bound: 4550.9828625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4552.1779534, upper bound: 4552.1779534
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4552.1779534, upper bound: 4552.1779534
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.7177194, upper bound: 4551.7177194
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.7177194, upper bound: 4551.7177194
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.1290439, upper bound: 4551.1290439
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.1290439, upper bound: 4551.1290439
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.52
Output dim: 0, lower bound: -4550.9127582, upper bound: 4550.9127784
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.52
Output dim: 0, lower bound: -4550.9127582, upper bound: 4550.9127582
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4552.1786120, upper bound: 4552.1785837
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4552.1785837, upper bound: 4552.1785837
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.2656369, upper bound: 4551.2734672
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.2656369, upper bound: 4551.2734672
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4552.1791231, upper bound: 4552.1791096
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4552.1791231, upper bound: 4552.1827321
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4552.1777914, upper bound: 4552.1810945
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4552.1777952, upper bound: 4552.1810199
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4552.1785432, upper bound: 4552.1785288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4552.1785445, upper bound: 4552.1785288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4552.0758610, upper bound: 4552.0759294
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4552.0758610, upper bound: 4552.0759286
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4552.2444294, upper bound: 4552.2444123
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4552.2444300, upper bound: 4552.2444123
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.7548688, upper bound: 4551.7548688
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.7548688, upper bound: 4551.7548688
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4552.0720295, upper bound: 4552.0720295
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4552.0720295, upper bound: 4552.0720295
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.2055892, upper bound: 4551.2117781
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.2055892, upper bound: 4551.2117781
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4552.1830724, upper bound: 4552.1893609
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4552.1830724, upper bound: 4552.1893612
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.9019164, upper bound: 4551.9019785
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.9019164, upper bound: 4551.9053101
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.8610584, upper bound: 4551.8656602
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.8610584, upper bound: 4551.8668863
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.7271371, upper bound: 4551.7271371
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.7271371, upper bound: 4551.7322659
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.8612279, upper bound: 4551.8655644
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.8610584, upper bound: 4551.8610584
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.9287906, upper bound: 4551.9331253
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.52
Output dim: 0, lower bound: -4551.9287906, upper bound: 4551.9331252

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0618276, upper bound: 4552.0618946
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0618276, upper bound: 4552.0618548
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1651622, upper bound: 4552.1651622
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1651622, upper bound: 4552.1651622
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953
1: -2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344
2: -3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344
3: -1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836
4: -3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0687996, upper bound: 4552.0687996
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0687996, upper bound: 4552.0687996
time: 0.77 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 0, lower bound: -4552.0618276, upper bound: 4552.0618946
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 0, lower bound: -4552.0618276, upper bound: 4552.0618548
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 0, lower bound: -4552.1651622, upper bound: 4552.1651622
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 0, lower bound: -4552.1651622, upper bound: 4552.1651622
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 0, lower bound: -4552.0687996, upper bound: 4552.0687996
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 0, lower bound: -4552.0687996, upper bound: 4552.0687996
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4552.1779534, upper bound: 4552.1779534
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4552.1786120, upper bound: 4552.1785837
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4552.1785837, upper bound: 4552.1785837
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4552.1791231, upper bound: 4552.1791096
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4552.1791231, upper bound: 4552.1827321
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4552.1777914, upper bound: 4552.1810945
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4552.1777952, upper bound: 4552.1810199
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4552.1785432, upper bound: 4552.1785288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4552.1785445, upper bound: 4552.1785288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4552.0758610, upper bound: 4552.0759294
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4552.0758610, upper bound: 4552.0759286
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4552.2444294, upper bound: 4552.2444123
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4552.2444300, upper bound: 4552.2444123
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4552.0720295, upper bound: 4552.0720295
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4552.0720295, upper bound: 4552.0720295
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4552.1830724, upper bound: 4552.1893609
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4552.1830724, upper bound: 4552.1893612
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4551.9019164, upper bound: 4551.9019785
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4551.9019164, upper bound: 4551.9053101
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4551.8610584, upper bound: 4551.8656602
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4551.8610584, upper bound: 4551.8668863
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4551.8612279, upper bound: 4551.8655644
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4551.8610584, upper bound: 4551.8610584
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4551.9287906, upper bound: 4551.9331253
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 0, lower bound: -4551.9287906, upper bound: 4551.9331252
Binary search (step 7): status=Status.UNKNOWN, low=0.9609375, high=0.9648438, mid=0.9648438, abs_max=5687.5751953125
rel_dist={0: [-4552.289001211694, 4552.289001211693]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.9609375
execution time: 1112.10 seconds
