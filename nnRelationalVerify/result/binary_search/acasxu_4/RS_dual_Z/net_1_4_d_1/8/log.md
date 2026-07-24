## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_4.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 46.318565822800004


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495)
1: (-17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504)
2: (-17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372)
3: (-22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624)
4: (-20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585)

## BASE Result
execution time: IAR + LP analysis = 2.46 + 1.88 = 4.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -46.4113886, upper bound: 46.4113886


# Binary Search by BASE starts (time budget: 1195.65 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=55.00176239013672
rel_dist={3: [-46.41138857101316, 46.41138857101315]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=55.00176239013672
rel_dist={3: [-46.411366576283044, 46.411366576283044]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=55.00176239013672
rel_dist={3: [-46.41123857032521, 46.41123857032903]}

## Binary search (step 3) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=55.00176239013672
rel_dist={3: [-46.410039219938405, 46.4100392199384]}

## Binary search (step 4) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=55.00176239013672
rel_dist={3: [-46.40791696364876, 46.40791696367562]}

## Binary search (step 5) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=55.00176239013672
rel_dist={3: [-46.40672265181321, 46.40672265182802]}

## Binary search (step 6) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=55.00176239013672
rel_dist={3: [-46.40611311757378, 46.406113117581526]}

## Binary search (step 7) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=55.00176239013672
rel_dist={3: [-46.40580581063882, 46.40580581064276]}

## Binary search (step 8) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=55.00176239013672
rel_dist={3: [-46.405651916648736, 46.40565189290891]}

## Binary search (step 9) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=55.00176239013672
rel_dist={3: [-46.405574448472976, 46.4055744584641]}

## Binary search (step 10) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=55.00176239013672
rel_dist={3: [-46.40553548314551, 46.40553551008645]}

## Binary search (step 11) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=55.00176239013672
rel_dist={3: [-46.40551600158139, 46.405516036735335]}

## Binary search (step 12) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=55.00176239013672
rel_dist={3: [-46.40550626293548, 46.4055063016896]}

## Binary search (step 13) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=55.00176239013672
rel_dist={3: [-46.405501397303595, 46.405501397303524]}

## Binary search (step 14) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=55.00176239013672
rel_dist={3: [-46.40549902064443, 46.40549902136124]}

## Binary search (step 15) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=55.00176239013672
rel_dist={3: [-46.40549778989501, 46.4054978543421]}

## Binary search (step 16) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=55.00176239013672
rel_dist={3: [-46.40549734791982, 46.405497434778894]}

## Binary Search Result
Binary search time: 75.09 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1120.56 seconds

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4113240, upper bound: 46.4074624
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4074624, upper bound: 46.4113240
time: 0.60 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 3, lower bound: -46.4113240, upper bound: 46.4074624
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 3, lower bound: -46.4074624, upper bound: 46.4113240

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4074355, upper bound: 46.4074355
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4074355, upper bound: 46.4074624
time: 0.79 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4074355, upper bound: 46.4112779
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4074355, upper bound: 46.4112779
time: 1.01 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.23 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.23
Output dim: 3, lower bound: -46.4074355, upper bound: 46.4074355
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.23
Output dim: 3, lower bound: -46.4074355, upper bound: 46.4074624
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.23
Output dim: 3, lower bound: -46.4074355, upper bound: 46.4112779
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.23
Output dim: 3, lower bound: -46.4074355, upper bound: 46.4112779

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4087899, upper bound: 46.4047632
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4047602, upper bound: 46.4047652
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4089062, upper bound: 46.4047602
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4047632, upper bound: 46.4047853
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4047853, upper bound: 46.4089062
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4047602, upper bound: 46.4089062
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4047652, upper bound: 46.4087899
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4047632, upper bound: 46.4087899
time: 0.77 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.35 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 3, lower bound: -46.4087899, upper bound: 46.4047632
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 3, lower bound: -46.4047602, upper bound: 46.4047652
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 3, lower bound: -46.4089062, upper bound: 46.4047602
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 3, lower bound: -46.4047632, upper bound: 46.4047853
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 3, lower bound: -46.4047853, upper bound: 46.4089062
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 3, lower bound: -46.4047602, upper bound: 46.4089062
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 3, lower bound: -46.4047652, upper bound: 46.4087899
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 3, lower bound: -46.4047632, upper bound: 46.4087899

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996030, upper bound: 46.3996034
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4039308, upper bound: 46.3996034
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4039308, upper bound: 46.3996034
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996029, upper bound: 46.3996034
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996034, upper bound: 46.3996029
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996034, upper bound: 46.3996029
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4039149, upper bound: 46.3996030
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996034, upper bound: 46.3996030
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996030, upper bound: 46.4037056
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996030, upper bound: 46.4039149
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996029, upper bound: 46.4036653
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996029, upper bound: 46.4039149
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996034, upper bound: 46.4039308
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996034, upper bound: 46.4039308
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996034, upper bound: 46.4039308
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996034, upper bound: 46.4039308
time: 0.97 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 3, lower bound: -46.3996030, upper bound: 46.3996034
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 3, lower bound: -46.4039308, upper bound: 46.3996034
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 3, lower bound: -46.4039308, upper bound: 46.3996034
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 3, lower bound: -46.3996029, upper bound: 46.3996034
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 3, lower bound: -46.3996034, upper bound: 46.3996029
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 3, lower bound: -46.3996034, upper bound: 46.3996029
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 3, lower bound: -46.4039149, upper bound: 46.3996030
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 3, lower bound: -46.3996034, upper bound: 46.3996030
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 3, lower bound: -46.3996030, upper bound: 46.4037056
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 3, lower bound: -46.3996030, upper bound: 46.4039149
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 3, lower bound: -46.3996029, upper bound: 46.4036653
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 3, lower bound: -46.3996029, upper bound: 46.4039149
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 3, lower bound: -46.3996034, upper bound: 46.4039308
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 3, lower bound: -46.3996034, upper bound: 46.4039308
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 3, lower bound: -46.3996034, upper bound: 46.4039308
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.42
Output dim: 3, lower bound: -46.3996034, upper bound: 46.4039308

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999252, upper bound: 46.3993646
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4002932
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4029667
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4002932
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4031679
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4002932
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4029606
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4002932
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4031679
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4032006
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4032006
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3999252
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4032006
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3999252
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4032006
time: 0.85 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3999252, upper bound: 46.3993646
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4002932
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4029667
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4002932
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4031679
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4002932
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4029606
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4002932
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4031679
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4032006
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4032006
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3999252
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4032006
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3999252
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4032006

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990249, upper bound: 46.3988380
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988463, upper bound: 46.3988380
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988463, upper bound: 46.3988380
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990211, upper bound: 46.3988380
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990211, upper bound: 46.3988380
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988463, upper bound: 46.3988380
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988463, upper bound: 46.3988380
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990249, upper bound: 46.3988380
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990211, upper bound: 46.3988380
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990211, upper bound: 46.3988380
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988974, upper bound: 46.3988380
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988974, upper bound: 46.3988380
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990807, upper bound: 46.3988380
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988974, upper bound: 46.3988380
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3989848, upper bound: 46.3988380
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988974, upper bound: 46.3988380
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988974, upper bound: 46.3988380
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988974
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988974
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3989848
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3989848
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988974
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988974
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990807
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990807
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988974
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988974
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3989848
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3989848
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988974
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988974
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990807
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990807
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990211
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990211
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990249
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990249
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988463
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988463
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990211
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990211
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988463
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988463
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990249
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990249
time: 0.82 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3990249, upper bound: 46.3988380
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988463, upper bound: 46.3988380
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988463, upper bound: 46.3988380
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3990211, upper bound: 46.3988380
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3990211, upper bound: 46.3988380
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988463, upper bound: 46.3988380
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988463, upper bound: 46.3988380
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3990249, upper bound: 46.3988380
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3990211, upper bound: 46.3988380
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3990211, upper bound: 46.3988380
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988974, upper bound: 46.3988380
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988974, upper bound: 46.3988380
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3990807, upper bound: 46.3988380
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988974, upper bound: 46.3988380
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3989848, upper bound: 46.3988380
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988974, upper bound: 46.3988380
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988974, upper bound: 46.3988380
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988974
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988974
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3989848
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3989848
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988974
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988974
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990807
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990807
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988974
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988974
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3989848
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3989848
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988974
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988974
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990807
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990807
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990211
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990211
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990249
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990249
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988463
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988463
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990211
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990211
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988463
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988463
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990249
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990249

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3965321, upper bound: 46.3962936
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3962936, upper bound: 46.3962936
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966303, upper bound: 46.3962936
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3962936, upper bound: 46.3962936
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3962936, upper bound: 46.3962936
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3962936, upper bound: 46.3962936
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966303, upper bound: 46.3962936
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3962936, upper bound: 46.3962936
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726392, upper bound: 46.3726379
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726392, upper bound: 46.3726379
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3964958, upper bound: 46.3962936
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3962936, upper bound: 46.3962936
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3965727, upper bound: 46.3962936
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3962936, upper bound: 46.3962936
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726392, upper bound: 46.3726379
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726392, upper bound: 46.3726379
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3965329, upper bound: 46.3962936
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3962936, upper bound: 46.3962936
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3962936, upper bound: 46.3962936
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3962936, upper bound: 46.3962936
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.70 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=55.00176239013672
rel_dist={3: [-46.41138857101316, 46.41138857101315]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4074622, upper bound: 46.4074622
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4074622, upper bound: 46.4113062
time: 0.81 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 3, lower bound: -46.4074622, upper bound: 46.4074622
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 3, lower bound: -46.4074622, upper bound: 46.4113062

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4089048, upper bound: 46.4047627
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4089048, upper bound: 46.4047836
time: 0.73 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4047836, upper bound: 46.4089048
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4047627, upper bound: 46.4089048
time: 0.77 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.32 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.32
Output dim: 3, lower bound: -46.4089048, upper bound: 46.4047627
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.32
Output dim: 3, lower bound: -46.4089048, upper bound: 46.4047836
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.32
Output dim: 3, lower bound: -46.4047836, upper bound: 46.4089048
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.32
Output dim: 3, lower bound: -46.4047627, upper bound: 46.4089048

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4047836, upper bound: 46.4047627
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4047646, upper bound: 46.4047601
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4047601, upper bound: 46.4047646
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4047627, upper bound: 46.4047836
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4047836, upper bound: 46.4089048
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4047646, upper bound: 46.4087894
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4047601, upper bound: 46.4089048
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4047627, upper bound: 46.4087894
time: 0.71 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.94 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.94
Output dim: 3, lower bound: -46.4047836, upper bound: 46.4047627
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.94
Output dim: 3, lower bound: -46.4047646, upper bound: 46.4047601
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.94
Output dim: 3, lower bound: -46.4047601, upper bound: 46.4047646
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.94
Output dim: 3, lower bound: -46.4047627, upper bound: 46.4047836
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.94
Output dim: 3, lower bound: -46.4047836, upper bound: 46.4089048
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.94
Output dim: 3, lower bound: -46.4047646, upper bound: 46.4087894
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.94
Output dim: 3, lower bound: -46.4047601, upper bound: 46.4089048
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.94
Output dim: 3, lower bound: -46.4047627, upper bound: 46.4087894

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996030, upper bound: 46.3996034
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4039291, upper bound: 46.3996034
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996034, upper bound: 46.3996021
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4036638, upper bound: 46.3996021
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996021, upper bound: 46.3996034
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996021, upper bound: 46.3996034
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996034, upper bound: 46.3996030
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996034, upper bound: 46.3996030
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996030, upper bound: 46.4037031
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996030, upper bound: 46.4039133
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996030, upper bound: 46.4039291
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996034, upper bound: 46.4039291
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996021, upper bound: 46.4036638
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996021, upper bound: 46.4039133
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996034, upper bound: 46.4039291
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3996021, upper bound: 46.4039291
time: 0.77 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3996030, upper bound: 46.3996034
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.4039291, upper bound: 46.3996034
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3996034, upper bound: 46.3996021
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.4036638, upper bound: 46.3996021
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3996021, upper bound: 46.3996034
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3996021, upper bound: 46.3996034
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3996034, upper bound: 46.3996030
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3996034, upper bound: 46.3996030
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3996030, upper bound: 46.4037031
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3996030, upper bound: 46.4039133
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3996030, upper bound: 46.4039291
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3996034, upper bound: 46.4039291
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3996021, upper bound: 46.4036638
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3996021, upper bound: 46.4039133
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3996034, upper bound: 46.4039291
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -46.3996021, upper bound: 46.4039291

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4002903, upper bound: 46.3993646
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4002903
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4029651
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4002903
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4031658
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4031981
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4031981
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4002903
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4029584
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4002903
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4031658
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3999237
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4031981
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3999237
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4031981
time: 1.10 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.4002903, upper bound: 46.3993646
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4002903
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4029651
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4002903
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4031658
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4031981
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3993646
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4031981
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4002903
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4029584
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4002903
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4031658
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3999237
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4031981
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.3999237
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.83
Output dim: 3, lower bound: -46.3993646, upper bound: 46.4031981

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988463, upper bound: 46.3988362
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988463, upper bound: 46.3988362
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988463, upper bound: 46.3988362
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990238, upper bound: 46.3988362
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990203, upper bound: 46.3988362
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990427, upper bound: 46.3988362
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988974, upper bound: 46.3988362
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988974, upper bound: 46.3988362
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988974, upper bound: 46.3988362
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988974
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988974
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3989848
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3989848
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988974
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988974
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990427
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990427
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990203
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990203
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990238
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990238
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988974
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988974
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3989848
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3989848
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988974
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988974
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990427
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990427
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988463
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988463
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990203
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990203
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988463
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988463
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990238
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990238
time: 0.96 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.70 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988463, upper bound: 46.3988362
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988463, upper bound: 46.3988362
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988463, upper bound: 46.3988362
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3990238, upper bound: 46.3988362
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3990203, upper bound: 46.3988362
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3990427, upper bound: 46.3988362
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988974, upper bound: 46.3988362
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988974, upper bound: 46.3988362
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988974, upper bound: 46.3988362
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988974
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988974
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3989848
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3989848
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988974
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988974
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990427
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990427
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990203
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990203
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988362
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990238
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990238
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988974
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988974
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3989848
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3989848
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988974
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988974
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990427
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990427
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988463
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988463
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990203
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990203
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988463
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3988463
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990238
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 3, lower bound: -46.3988362, upper bound: 46.3990238

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3968741, upper bound: 46.3966491
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3968741, upper bound: 46.3966491
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966874, upper bound: 46.3966491
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3968741, upper bound: 46.3966491
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3968741, upper bound: 46.3966491
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966874, upper bound: 46.3966491
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966874, upper bound: 46.3966491
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3968727, upper bound: 46.3966491
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3967568, upper bound: 46.3966491
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3968445, upper bound: 46.3966491
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3968741, upper bound: 46.3966491
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3968741, upper bound: 46.3966491
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3968741, upper bound: 46.3966491
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3968741, upper bound: 46.3966491
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3968727, upper bound: 46.3966491
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3968727, upper bound: 46.3966491
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3967568, upper bound: 46.3966491
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3968445, upper bound: 46.3966491
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3968445, upper bound: 46.3966491
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3967568
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3967568
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3968445
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3966491, upper bound: 46.3966491
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.56 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=55.00176239013672
rel_dist={3: [-46.411366576283044, 46.411366576283044]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4071680, upper bound: 46.4071680
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4071680, upper bound: 46.4111799
time: 0.77 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.85 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.85
Output dim: 3, lower bound: -46.4071680, upper bound: 46.4071680
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.85
Output dim: 3, lower bound: -46.4071680, upper bound: 46.4111799

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4044776
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4044776
time: 0.67 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4087891
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4087891
time: 0.77 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.12
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4044776
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.12
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4044776
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.12
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4087891
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.12
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4087891

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4044776
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4044776
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4044776
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4044776
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4087891
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4086209
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4087891
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4086209
time: 0.86 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.47 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4044776
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4044776
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4044776
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4044776
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4087891
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4086209
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4087891
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.47
Output dim: 3, lower bound: -46.4044776, upper bound: 46.4086209

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994151, upper bound: 46.3994168
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994145, upper bound: 46.3994168
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994168, upper bound: 46.3994145
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994168, upper bound: 46.3994145
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994145, upper bound: 46.3994168
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994145, upper bound: 46.3994168
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994145, upper bound: 46.3994151
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994145, upper bound: 46.3994151
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994151, upper bound: 46.4035718
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994151, upper bound: 46.4037382
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994151, upper bound: 46.4037624
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994168, upper bound: 46.4037626
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994145, upper bound: 46.4035350
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994145, upper bound: 46.4037382
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994168, upper bound: 46.4037624
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3994168, upper bound: 46.4037626
time: 0.99 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.36 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -46.3994151, upper bound: 46.3994168
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -46.3994145, upper bound: 46.3994168
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -46.3994168, upper bound: 46.3994145
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -46.3994168, upper bound: 46.3994145
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -46.3994145, upper bound: 46.3994168
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -46.3994145, upper bound: 46.3994168
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -46.3994145, upper bound: 46.3994151
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -46.3994145, upper bound: 46.3994151
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -46.3994151, upper bound: 46.4035718
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -46.3994151, upper bound: 46.4037382
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -46.3994151, upper bound: 46.4037624
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -46.3994168, upper bound: 46.4037626
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -46.3994145, upper bound: 46.4035350
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -46.3994145, upper bound: 46.4037382
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -46.3994168, upper bound: 46.4037624
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -46.3994168, upper bound: 46.4037626

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4002127
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4029157
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4002127
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4030101
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4030487
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4030487
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4002127
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4029108
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4002127
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4030101
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3998421
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4030487
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3998421
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4030487
time: 0.83 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4002127
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4029157
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4002127
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4030101
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4030487
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3991351
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4030487
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4002127
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4029108
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4002127
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4030101
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3998421
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4030487
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.3998421
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.28
Output dim: 3, lower bound: -46.3991351, upper bound: 46.4030487

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988402, upper bound: 46.3986548
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988402, upper bound: 46.3986548
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986616, upper bound: 46.3986548
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986616, upper bound: 46.3986548
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988361, upper bound: 46.3986548
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986616, upper bound: 46.3986548
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986616, upper bound: 46.3986548
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3989322, upper bound: 46.3986548
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3987127, upper bound: 46.3986548
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3987127, upper bound: 46.3986548
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3987127, upper bound: 46.3986548
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988402, upper bound: 46.3986548
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988361, upper bound: 46.3986548
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3989322, upper bound: 46.3986548
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3989322, upper bound: 46.3986548
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3987127, upper bound: 46.3986548
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3987127, upper bound: 46.3986548
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3987997, upper bound: 46.3986548
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3987997, upper bound: 46.3986548
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3987127, upper bound: 46.3986548
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3987127, upper bound: 46.3986548
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987127
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987127
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987997
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987997
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987127
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987127
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3989322
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3989322
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3988361
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3988361
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3988402
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3988402
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987127
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987127
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987997
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987997
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987127
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987127
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3989322
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3989322
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986616
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986616
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3988361
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3988361
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986616
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986616
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3988402
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3988402
time: 0.95 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.71 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3988402, upper bound: 46.3986548
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3988402, upper bound: 46.3986548
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986616, upper bound: 46.3986548
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986616, upper bound: 46.3986548
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3988361, upper bound: 46.3986548
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986616, upper bound: 46.3986548
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986616, upper bound: 46.3986548
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3989322, upper bound: 46.3986548
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3987127, upper bound: 46.3986548
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3987127, upper bound: 46.3986548
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3987127, upper bound: 46.3986548
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3988402, upper bound: 46.3986548
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3988361, upper bound: 46.3986548
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3989322, upper bound: 46.3986548
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3989322, upper bound: 46.3986548
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3987127, upper bound: 46.3986548
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3987127, upper bound: 46.3986548
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3987997, upper bound: 46.3986548
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3987997, upper bound: 46.3986548
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3987127, upper bound: 46.3986548
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3987127, upper bound: 46.3986548
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987127
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987127
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987997
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987997
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987127
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987127
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3989322
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3989322
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3988361
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3988361
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986548
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3988402
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3988402
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987127
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987127
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987997
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987997
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987127
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3987127
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3989322
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3989322
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986616
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986616
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3988361
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3988361
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986616
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3986616
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3988402
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -46.3986548, upper bound: 46.3988402
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=55.00176239013672
rel_dist={3: [-46.41123857032521, 46.41123857032903]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1120.98 seconds
