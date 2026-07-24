## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 466.672919624136


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641)
1: (-154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164)
2: (-136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744)
3: (-210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305)
4: (-164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.46 + 1.90 = 3.36 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -466.6775864, upper bound: 466.6775864

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742992, upper bound: 466.6742992
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742992, upper bound: 466.6743005
time: 0.80 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.58 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -466.6742992, upper bound: 466.6742992
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -466.6742992, upper bound: 466.6743005

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742583, upper bound: 466.6742583
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742583, upper bound: 466.6742755
time: 0.71 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742583, upper bound: 466.6742595
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742583, upper bound: 466.6742764
time: 0.98 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.15 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -466.6742583, upper bound: 466.6742583
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -466.6742583, upper bound: 466.6742755
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -466.6742583, upper bound: 466.6742595
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -466.6742583, upper bound: 466.6742764

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742536, upper bound: 466.6742506
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742376, upper bound: 466.6742257
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742370, upper bound: 466.6742649
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742370, upper bound: 466.6742430
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742430, upper bound: 466.6742376
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742649, upper bound: 466.6742370
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742257, upper bound: 466.6742539
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742257, upper bound: 466.6742536
time: 0.85 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.00 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -466.6742536, upper bound: 466.6742506
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -466.6742376, upper bound: 466.6742257
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -466.6742370, upper bound: 466.6742649
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -466.6742370, upper bound: 466.6742430
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -466.6742430, upper bound: 466.6742376
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -466.6742649, upper bound: 466.6742370
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -466.6742257, upper bound: 466.6742539
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -466.6742257, upper bound: 466.6742536

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739768, upper bound: 466.6739872
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739911, upper bound: 466.6740003
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739913, upper bound: 466.6739682
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739772, upper bound: 466.6739754
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739768, upper bound: 466.6740019
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739768, upper bound: 466.6740200
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739772, upper bound: 466.6739874
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739772, upper bound: 466.6739980
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739682, upper bound: 466.6739772
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739682, upper bound: 466.6739894
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739682, upper bound: 466.6739768
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6740019, upper bound: 466.6739888
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739682, upper bound: 466.6739913
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739682, upper bound: 466.6740086
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739872, upper bound: 466.6739911
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739682, upper bound: 466.6740083
time: 1.00 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.22 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -466.6739768, upper bound: 466.6739872
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -466.6739911, upper bound: 466.6740003
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -466.6739913, upper bound: 466.6739682
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -466.6739772, upper bound: 466.6739754
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -466.6739768, upper bound: 466.6740019
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -466.6739768, upper bound: 466.6740200
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -466.6739772, upper bound: 466.6739874
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -466.6739772, upper bound: 466.6739980
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -466.6739682, upper bound: 466.6739772
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -466.6739682, upper bound: 466.6739894
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -466.6739682, upper bound: 466.6739768
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -466.6740019, upper bound: 466.6739888
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -466.6739682, upper bound: 466.6739913
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -466.6739682, upper bound: 466.6740086
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -466.6739872, upper bound: 466.6739911
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -466.6739682, upper bound: 466.6740083

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739980, upper bound: 466.6739830
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739670, upper bound: 466.6739872
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739761
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739686, upper bound: 466.6740002
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6740078, upper bound: 466.6739669
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739688, upper bound: 466.6739669
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739671, upper bound: 466.6739669
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739688, upper bound: 466.6739754
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739670, upper bound: 466.6739910
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6740019
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739768, upper bound: 466.6739782
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739670, upper bound: 466.6740195
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739894, upper bound: 466.6739795
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739671, upper bound: 466.6739874
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739671, upper bound: 466.6739669
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739671, upper bound: 466.6739980
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739713
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739772
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739671
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739894
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739708
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739768
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739830, upper bound: 466.6739670
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739885
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739812
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739913
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739688
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6740078
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739754, upper bound: 466.6739810
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739911
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739686
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739830, upper bound: 466.6740075
time: 0.92 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.72 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739980, upper bound: 466.6739830
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739670, upper bound: 466.6739872
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739761
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739686, upper bound: 466.6740002
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6740078, upper bound: 466.6739669
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739688, upper bound: 466.6739669
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739671, upper bound: 466.6739669
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739688, upper bound: 466.6739754
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739670, upper bound: 466.6739910
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6740019
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739768, upper bound: 466.6739782
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739670, upper bound: 466.6740195
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739894, upper bound: 466.6739795
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739671, upper bound: 466.6739874
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739671, upper bound: 466.6739669
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739671, upper bound: 466.6739980
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739713
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739772
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739671
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739894
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739708
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739768
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739830, upper bound: 466.6739670
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739885
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739812
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739913
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739688
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6740078
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739754, upper bound: 466.6739810
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739911
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739669, upper bound: 466.6739686
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -466.6739830, upper bound: 466.6740075

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738967, upper bound: 466.6739081
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739109
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739118
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739147
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738967, upper bound: 466.6739038
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739045
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738967, upper bound: 466.6739165
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739286
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738970, upper bound: 466.6738952
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738970, upper bound: 466.6738952
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738970, upper bound: 466.6738952
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739121, upper bound: 466.6738952
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739070, upper bound: 466.6738952
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739170
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739180
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739256
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739276
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738982, upper bound: 466.6739041
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739068
time: 2.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738982, upper bound: 466.6739359
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739460
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739060
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739067
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739138
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739143
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738985, upper bound: 466.6738952
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
time: 2.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739188
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739254
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738985
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739046
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739138, upper bound: 466.6738952
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738988
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739160
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738970, upper bound: 466.6738952
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738982
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739044
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738988
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739155
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739042
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739070
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739121
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739163
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738970
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739123
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739338
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739059
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739070
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739138
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739161
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739081, upper bound: 466.6738967
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739188
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739335
time: 0.68 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.96 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738967, upper bound: 466.6739081
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739109
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739118
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739147
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738967, upper bound: 466.6739038
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739045
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738967, upper bound: 466.6739165
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739286
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738970, upper bound: 466.6738952
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738970, upper bound: 466.6738952
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738970, upper bound: 466.6738952
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6739121, upper bound: 466.6738952
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6739070, upper bound: 466.6738952
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739170
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739180
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739256
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739276
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738982, upper bound: 466.6739041
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739068
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738982, upper bound: 466.6739359
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739460
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739060
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739067
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739138
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739143
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738985, upper bound: 466.6738952
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739188
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739254
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738985
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739046
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6739138, upper bound: 466.6738952
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738988
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739160
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738970, upper bound: 466.6738952
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738982
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739044
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738988
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739155
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739042
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739070
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739163
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738970
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739123
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739338
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739059
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739070
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739138
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739161
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6738952
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6739081, upper bound: 466.6738967
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739188
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.96
Output dim: 0, lower bound: -466.6738952, upper bound: 466.6739335

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738820, upper bound: 466.6739042
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6739072
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738847
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738820, upper bound: 466.6739097
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6739120
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738933
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738919, upper bound: 466.6738996
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739002, upper bound: 466.6738815
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6739005
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738820
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738820, upper bound: 466.6739163
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6739263
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738993
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738822, upper bound: 466.6738815
time: 3.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738822, upper bound: 466.6738815
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738999, upper bound: 466.6738815
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738943, upper bound: 466.6738815
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738822, upper bound: 466.6738815
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6739133
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738890
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6739145
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738934
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6739229
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738969
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6739250
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6739032
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738922, upper bound: 466.6739000
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6739028
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738831
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6739338
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6739445
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738886
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738903
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738997
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738979
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6739003
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6739006
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738833, upper bound: 466.6739051
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738825, upper bound: 466.6738815
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6739111
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6739019
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738825
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738833
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738930
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738931
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6739003, upper bound: 466.6738815
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738997, upper bound: 466.6738815
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738867
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6739013
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738999
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738824
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738914
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738922
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738831, upper bound: 466.6738815
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738831, upper bound: 466.6738867
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738815, upper bound: 466.6738815
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641
1: -154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164
2: -136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744
3: -210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305
4: -164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.36 + 416.83 = 420.19 seconds
