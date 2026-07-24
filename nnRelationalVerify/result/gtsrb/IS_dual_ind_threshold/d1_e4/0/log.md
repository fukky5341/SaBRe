## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 0)
Time budget: 1800 seconds
Split limit: 100
Threshold: 27.5202515007


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599)
1: (-25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7340012, 43.7340050)
2: (-21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4324722, 39.4324760)
3: (-25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306)
4: (-28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107)
5: (-23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462)
6: (-33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8840103, 51.8840065)
7: (-30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2755661, 51.2755661)
8: (-32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979)
9: (-26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770)
10: (-36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488)
11: (-27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483)
12: (-35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.8131561, 55.8131561)
13: (-40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602)
14: (-56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614)
15: (-28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330)
16: (-34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797)
17: (-51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588)
18: (-30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641)
19: (-17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2683220, 35.2683182)
20: (-20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881)
21: (-24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023)
22: (-25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976)
23: (-18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780)
24: (-22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526)
25: (-19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701)
26: (-33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132)
27: (-24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385)
28: (-18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0786629, 44.0786667)
29: (-25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006)
30: (-25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484)
31: (-21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679)
32: (-36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6482010, 49.6482048)
33: (-50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9246216, 69.9246216)
34: (-50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.8147430, 56.8147469)
35: (-42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5810356, 57.5810318)
36: (-41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5230408, 60.5230408)
37: (-55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8719940, 67.8720016)
38: (-52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299)
39: (-61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5897675, 79.5897675)
40: (-48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2941208, 59.2941132)
41: (-35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4173737, 53.4173737)
42: (-26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2487335, 38.2487411)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.69 + 82.77 = 85.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 31, lower bound: -27.5477993, upper bound: 27.5477993

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 547

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5419843, upper bound: 27.5373015
time: 65.67 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5419843, upper bound: 27.5419842
time: 57.92 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 123.71 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 123.71
Output dim: 31, lower bound: -27.5419843, upper bound: 27.5373015
IS_A2, status: Status.UNKNOWN, split count: 1, time: 123.71
Output dim: 31, lower bound: -27.5419843, upper bound: 27.5419842

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -47.0457344, 16.0430908, -47.0647049, 16.0490398, -63.0947723, 63.1077957
1: -25.2418976, 18.6867027, -25.2510223, 18.6949844, -43.7158737, 43.7181587
2: -21.4909859, 17.9399910, -21.5084686, 17.9439869, -39.4029541, 39.4182816
3: -24.9963379, 17.4989681, -25.0240402, 17.5067215, -42.5030594, 42.5230103
4: -28.3292694, 20.1887550, -28.3469620, 20.1934032, -48.5226746, 48.5357170
5: -23.6130848, 19.8653336, -23.6383228, 19.8723259, -43.4854126, 43.5036545
6: -33.7461472, 18.9016266, -33.7534332, 18.9075546, -51.8701477, 51.8719597
7: -30.2782707, 21.4503593, -30.3018036, 21.4560432, -51.2372208, 51.2567291
8: -32.2346573, 24.0310593, -32.2561569, 24.0374031, -56.2720604, 56.2872162
9: -26.7103252, 17.7173023, -26.7259083, 17.7243500, -44.4346771, 44.4432106
10: -36.5310516, 25.9863415, -36.5411568, 26.0109768, -62.5420303, 62.5274963
11: -27.4805012, 25.0685654, -27.4892159, 25.1022606, -52.5827637, 52.5577812
12: -35.1474762, 21.9859428, -35.1565590, 22.0359993, -55.7782669, 55.7355881
13: -40.0434990, 22.5323601, -40.0722809, 22.5428314, -62.5863304, 62.6046410
14: -56.4829140, 13.4845228, -56.5007629, 13.5264282, -70.0093384, 69.9852829
15: -28.2993279, 19.6026630, -28.3102036, 19.6105671, -47.9098969, 47.9128647
16: -34.9203606, 22.3906441, -34.9337540, 22.3963757, -57.3167343, 57.3243980
17: -51.8631096, 26.9820061, -51.8746338, 27.0383492, -78.9014587, 78.8566437
18: -30.0514126, 26.8600254, -30.0606575, 26.9040680, -56.9554825, 56.9206848
19: -17.7019463, 17.6383533, -17.7103615, 17.6475143, -35.2522430, 35.2502136
20: -20.0394154, 17.4697456, -20.0481186, 17.4854641, -37.5248795, 37.5178642
21: -24.4849091, 22.8025932, -24.4945488, 22.8290672, -47.3139763, 47.2971420
22: -25.5272598, 21.6558285, -25.5356007, 21.6808758, -47.2081375, 47.1914291
23: -18.7296333, 21.6605167, -18.7363129, 21.6761017, -40.4057350, 40.3968277
24: -22.4811802, 22.2763138, -22.4903603, 22.2959843, -44.7771645, 44.7666740
25: -19.4821663, 24.5758591, -19.4904137, 24.5981979, -44.0803642, 44.0662727
26: -33.8073006, 29.4401436, -33.8171158, 29.4952793, -63.3025818, 63.2572594
27: -24.1074829, 22.8418427, -24.1175709, 22.8581314, -46.9656143, 46.9594116
28: -18.7410927, 25.4510612, -18.7503586, 25.4558754, -44.0625916, 44.0662727
29: -25.6166325, 24.3103714, -25.6250973, 24.3387775, -49.9554100, 49.9354706
30: -25.7558517, 25.4038963, -25.7637100, 25.4297447, -51.1855965, 51.1676064
31: -21.9013443, 23.8004494, -21.9137154, 23.8140182, -45.7153625, 45.7141647
32: -36.0850601, 14.9704161, -36.0993080, 14.9780245, -49.6228180, 49.6299896
33: -50.8243675, 21.5007629, -50.8546257, 21.5092697, -69.8750610, 69.8988113
34: -50.5695305, 10.0486937, -50.5887680, 10.0572071, -56.7854004, 56.7939148
35: -42.9222145, 16.9610405, -42.9483871, 16.9671173, -57.5412827, 57.5608063
36: -41.9330254, 18.6401196, -41.9474335, 18.6462955, -60.4989166, 60.5075188
37: -55.7244034, 13.1818686, -55.7363129, 13.1921377, -67.8489380, 67.8504791
38: -52.8787613, 15.9190121, -52.8931122, 15.9332752, -68.8120346, 68.8121262
39: -61.7447166, 18.2048931, -61.7695389, 18.2100754, -79.5503159, 79.5707474
40: -48.4160233, 11.8155355, -48.4354973, 11.8201342, -59.2633514, 59.2787704
41: -35.8978119, 18.6594810, -35.9089394, 18.6663113, -53.3971252, 53.4021378
42: -26.5182533, 13.2142944, -26.5247231, 13.2287474, -38.2339439, 38.2247849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=153, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 547

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 697

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5391016, upper bound: 27.5329856
time: 69.48 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5376732, upper bound: 27.5329856
time: 84.51 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -47.1169891, 16.0694160, -47.0584869, 16.0431786, -63.1601677, 63.1279030
1: -25.2744846, 18.7200966, -25.2471714, 18.6898079, -43.7437897, 43.7485542
2: -21.5338364, 18.0324097, -21.5074425, 17.9412956, -39.4422798, 39.5200195
3: -25.0421066, 17.6848907, -25.0265999, 17.5040150, -42.5461197, 42.7114906
4: -28.3780537, 20.2911568, -28.3441315, 20.1913528, -48.5694046, 48.6352882
5: -23.6680565, 20.0164108, -23.6406174, 19.8696556, -43.5377121, 43.6570282
6: -33.7960434, 18.9243832, -33.7504501, 18.8998280, -51.9109650, 51.9094734
7: -30.3407555, 21.5035191, -30.3014412, 21.4518242, -51.2917328, 51.3178024
8: -32.2861137, 24.0975361, -32.2576408, 24.0335197, -56.3196335, 56.3551788
9: -26.7457523, 17.8185883, -26.7135887, 17.7225590, -44.4683113, 44.5321770
10: -36.7268944, 26.0485497, -36.5393753, 26.0055065, -62.7323990, 62.5879250
11: -27.7378025, 25.1196880, -27.4832363, 25.1060829, -52.8438873, 52.6029243
12: -35.3851051, 22.0773201, -35.1537704, 22.0416508, -56.0256119, 55.8190384
13: -40.0681458, 22.7064190, -40.0522728, 22.5394077, -62.6075516, 62.7586899
14: -56.7985458, 13.5458832, -56.4973373, 13.5314312, -70.3299789, 70.0432205
15: -28.3087807, 19.6725178, -28.2926083, 19.6082535, -47.9170341, 47.9651260
16: -35.0393639, 22.4373264, -34.9297333, 22.3927383, -57.4321022, 57.3670578
17: -52.1469116, 27.0688057, -51.8713531, 27.0433941, -79.1903076, 78.9401550
18: -30.3176575, 26.9241142, -30.0573711, 26.9080963, -57.2257538, 56.9814835
19: -17.8304596, 17.6590824, -17.7078629, 17.6480789, -35.3881760, 35.2682495
20: -20.1456680, 17.5004539, -20.0467262, 17.4859619, -37.6316299, 37.5471802
21: -24.6846218, 22.8432236, -24.4926224, 22.8320885, -47.5167084, 47.3358459
22: -25.6347027, 21.6968994, -25.5300045, 21.6810799, -47.3157806, 47.2240524
23: -18.8706989, 21.6926079, -18.7339859, 21.6750450, -40.5457458, 40.4265938
24: -22.6153774, 22.2859306, -22.4837875, 22.2825985, -44.8979759, 44.7697182
25: -19.6031532, 24.6105156, -19.4848557, 24.5988274, -44.2019806, 44.0953712
26: -34.0886116, 29.5376816, -33.8116112, 29.5010529, -63.5896645, 63.3492928
27: -24.2041626, 22.8565369, -24.1137047, 22.8505821, -47.0547447, 46.9702415
28: -18.7931118, 25.4786568, -18.7458096, 25.4540787, -44.1144333, 44.0885658
29: -25.7490654, 24.3478260, -25.6185284, 24.3386421, -50.0877075, 49.9663544
30: -25.8866386, 25.4508209, -25.7583427, 25.4289627, -51.3156013, 51.2091637
31: -22.0736351, 23.8086967, -21.9112091, 23.8057404, -45.8793755, 45.7199059
32: -36.1331062, 15.0237713, -36.0947418, 14.9745426, -49.6686325, 49.6820679
33: -50.8955612, 21.6743355, -50.8560066, 21.5068359, -69.9371185, 70.0721588
34: -50.6043243, 10.1584120, -50.5884666, 10.0558472, -56.8084488, 56.9002190
35: -42.9673653, 17.1039143, -42.9482918, 16.9660530, -57.5816193, 57.7040482
36: -41.9547615, 18.7216702, -41.9379692, 18.6460648, -60.5199356, 60.5821838
37: -55.8251915, 13.2271147, -55.7307205, 13.1823521, -67.9398956, 67.8906555
38: -52.9278221, 15.9717274, -52.8888855, 15.9191151, -68.8469391, 68.8606110
39: -61.8162079, 18.3712692, -61.7617188, 18.2084675, -79.6213913, 79.7269592
40: -48.4750977, 11.8920059, -48.4293022, 11.8154945, -59.3156738, 59.3484573
41: -35.9387169, 18.7207432, -35.9058838, 18.6642265, -53.4344330, 53.4693604
42: -26.5825214, 13.2557898, -26.5230045, 13.2186985, -38.2934723, 38.2685089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=153, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 547

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 697

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5391016, upper bound: 27.5376731
time: 72.28 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5376732, upper bound: 27.5376731
time: 68.33 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 142.85 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 142.85
Output dim: 31, lower bound: -27.5391016, upper bound: 27.5329856
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 142.85
Output dim: 31, lower bound: -27.5376732, upper bound: 27.5329856
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 142.85
Output dim: 31, lower bound: -27.5391016, upper bound: 27.5376731
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 142.85
Output dim: 31, lower bound: -27.5376732, upper bound: 27.5376731

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -46.9909286, 16.0350533, -46.9500504, 16.0019741, -62.9929047, 62.9851036
1: -25.1989365, 18.6798267, -25.1709843, 18.6424408, -43.6170387, 43.6303558
2: -21.4641018, 17.9325256, -21.4549313, 17.9046059, -39.3347130, 39.3548012
3: -24.9735336, 17.4909782, -24.9791203, 17.4524803, -42.4260139, 42.4701004
4: -28.3070755, 20.1779995, -28.3022766, 20.1478920, -48.4549675, 48.4802780
5: -23.5817146, 19.8567085, -23.5782661, 19.8065357, -43.3882523, 43.4349747
6: -33.7236938, 18.8846760, -33.7010307, 18.8613873, -51.7547226, 51.7988548
7: -30.2304573, 21.4428062, -30.2099667, 21.3823700, -51.1129837, 51.1555252
8: -32.1909714, 24.0193157, -32.1707306, 23.9551640, -56.1461334, 56.1900482
9: -26.6913147, 17.7079964, -26.6852207, 17.6946602, -44.3859749, 44.3932190
10: -36.5151596, 25.9725151, -36.4880295, 25.9734459, -62.4886055, 62.4605446
11: -27.4427242, 25.0576782, -27.4027748, 25.0708313, -52.5135574, 52.4604530
12: -35.1364899, 21.9613876, -35.1189651, 21.9798622, -55.6845093, 55.6555710
13: -40.0162621, 22.5144997, -40.0161514, 22.4635010, -62.4797630, 62.5306511
14: -56.4211426, 13.4761209, -56.3667030, 13.4702339, -69.8913727, 69.8428268
15: -28.2820530, 19.5762062, -28.2600498, 19.5533390, -47.8353920, 47.8362579
16: -34.8816681, 22.3835869, -34.8490295, 22.3657684, -57.2474365, 57.2326164
17: -51.8084373, 26.9728432, -51.7584076, 26.9551735, -78.7636108, 78.7312469
18: -30.0331154, 26.8465157, -29.9929008, 26.8719578, -56.9050751, 56.8394165
19: -17.6834278, 17.6354961, -17.6710892, 17.6372089, -35.2187195, 35.2109642
20: -20.0068169, 17.4590588, -19.9815216, 17.4172745, -37.4240913, 37.4405823
21: -24.4611511, 22.7953110, -24.4457073, 22.8064556, -47.2676086, 47.2410202
22: -25.5087376, 21.6386318, -25.4767189, 21.6382942, -47.1470337, 47.1153488
23: -18.7084446, 21.6491547, -18.6788979, 21.6495609, -40.3580055, 40.3280525
24: -22.4657841, 22.2416344, -22.4162769, 22.2347832, -44.7005692, 44.6579132
25: -19.4672585, 24.5604267, -19.4443016, 24.5627728, -44.0300293, 44.0047302
26: -33.7892838, 29.4169121, -33.7651062, 29.4635315, -63.2528152, 63.1820183
27: -24.0867672, 22.8319931, -24.0581264, 22.8367996, -46.9235687, 46.8901215
28: -18.7198601, 25.4437008, -18.6971722, 25.4229355, -44.0045929, 44.0084457
29: -25.5945473, 24.3029613, -25.5666161, 24.3128433, -49.9073906, 49.8695755
30: -25.7379189, 25.3525314, -25.6785622, 25.3307934, -51.0687103, 51.0310936
31: -21.8807888, 23.7929020, -21.8615608, 23.7951927, -45.6759796, 45.6544647
32: -36.0671196, 14.9574003, -36.0602455, 14.9426880, -49.5323563, 49.5699234
33: -50.8108902, 21.4185562, -50.7044258, 21.3613338, -69.7186279, 69.6614227
34: -50.5546150, 9.9848423, -50.4654388, 9.9446192, -56.6634293, 56.5984802
35: -42.9092636, 16.9053116, -42.8297119, 16.8666611, -57.4264069, 57.3825951
36: -41.9231453, 18.6254749, -41.9215126, 18.6145630, -60.4525223, 60.4661713
37: -55.7080688, 13.1195889, -55.6072083, 13.0802031, -67.7244873, 67.6569824
38: -52.8413925, 15.9033756, -52.8071747, 15.8364220, -68.6778107, 68.7105484
39: -61.7323532, 18.1550980, -61.6584244, 18.1188412, -79.4469528, 79.4099274
40: -48.4028130, 11.7696047, -48.3619041, 11.7347441, -59.1733398, 59.1559448
41: -35.8826141, 18.6417999, -35.8601227, 18.6303139, -53.3312683, 53.3298798
42: -26.4948769, 13.1987190, -26.4741478, 13.1873741, -38.1052475, 38.1575394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=361, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 547

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1722

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5358212, upper bound: 27.5188281
time: 68.53 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5358212, upper bound: 27.5291147
time: 56.86 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -47.0391350, 16.0409164, -47.0540199, 16.0454826, -63.0846176, 63.0949364
1: -25.2379265, 18.6847763, -25.2445583, 18.6918564, -43.7085457, 43.7000618
2: -21.4882011, 17.9372158, -21.5039310, 17.9394989, -39.3953705, 39.3969040
3: -24.9940643, 17.4968510, -25.0203533, 17.5032883, -42.4973526, 42.5172043
4: -28.3261452, 20.1865406, -28.3418770, 20.1898270, -48.5159721, 48.5284195
5: -23.6101532, 19.8635731, -23.6335411, 19.8694687, -43.4796219, 43.4971161
6: -33.7347488, 18.8988037, -33.7352524, 18.9030056, -51.8545532, 51.8491325
7: -30.2742481, 21.4479141, -30.2952538, 21.4520245, -51.2287292, 51.2289810
8: -32.2310600, 24.0279865, -32.2502403, 24.0324554, -56.2635155, 56.2782288
9: -26.7026520, 17.7152729, -26.7137070, 17.7210808, -44.4237328, 44.4289780
10: -36.5285645, 25.9825516, -36.5372238, 26.0048027, -62.5333672, 62.5197754
11: -27.4757252, 25.0642624, -27.4815159, 25.0952835, -52.5710068, 52.5457764
12: -35.1412506, 21.9828758, -35.1464081, 22.0309982, -55.7431030, 55.7095871
13: -40.0379143, 22.5282726, -40.0632782, 22.5361938, -62.5741081, 62.5915527
14: -56.4751968, 13.4826994, -56.4882736, 13.5234489, -69.9986420, 69.9709702
15: -28.2956505, 19.5992279, -28.3041878, 19.6050453, -47.9006958, 47.9034157
16: -34.9126358, 22.3870773, -34.9211388, 22.3905735, -57.3032074, 57.3082161
17: -51.8571396, 26.9799385, -51.8649940, 27.0350342, -78.8921738, 78.8449326
18: -30.0483665, 26.8570404, -30.0557480, 26.8993378, -56.9477043, 56.9127884
19: -17.6983833, 17.6374130, -17.7045918, 17.6459923, -35.2444763, 35.2380486
20: -20.0356407, 17.4676075, -20.0419846, 17.4820137, -37.5176544, 37.5095901
21: -24.4804993, 22.7950172, -24.4874020, 22.8166580, -47.2971573, 47.2824173
22: -25.5214500, 21.6517944, -25.5261002, 21.6743011, -47.1955566, 47.1778946
23: -18.7269478, 21.6551895, -18.7319679, 21.6675434, -40.3944931, 40.3871574
24: -22.4777718, 22.2663212, -22.4848385, 22.2817955, -44.7595673, 44.7511597
25: -19.4783745, 24.5675926, -19.4843159, 24.5852165, -44.0635910, 44.0519104
26: -33.7886124, 29.4357052, -33.7861977, 29.4882240, -63.2768364, 63.2219009
27: -24.1000595, 22.8401031, -24.1053467, 22.8553829, -46.9554443, 46.9454498
28: -18.7359257, 25.4486408, -18.7420559, 25.4519463, -44.0528374, 44.0540237
29: -25.6108704, 24.3087177, -25.6155930, 24.3361130, -49.9469833, 49.9243088
30: -25.7525158, 25.3983688, -25.7583027, 25.4206753, -51.1731911, 51.1566696
31: -21.8963528, 23.7845554, -21.9056320, 23.7884254, -45.6847763, 45.6901855
32: -36.0716476, 14.9686995, -36.0775261, 14.9752808, -49.6042976, 49.6030121
33: -50.8206062, 21.4952278, -50.8485565, 21.5001984, -69.8203049, 69.8862076
34: -50.5658340, 10.0437298, -50.5827904, 10.0491066, -56.7082367, 56.7818527
35: -42.9184761, 16.9549942, -42.9423409, 16.9593353, -57.4844055, 57.5497398
36: -41.9257851, 18.6383133, -41.9354973, 18.6434441, -60.4885025, 60.4936295
37: -55.7212830, 13.1776152, -55.7312469, 13.1852083, -67.8194122, 67.8407440
38: -52.8740387, 15.9165058, -52.8854637, 15.9292011, -68.8032379, 68.8019714
39: -61.7404175, 18.1938210, -61.7625961, 18.1947594, -79.5106430, 79.5508423
40: -48.4135056, 11.8077002, -48.4314423, 11.8074055, -59.2385941, 59.2664719
41: -35.8868866, 18.6562614, -35.8908463, 18.6611137, -53.3759842, 53.3784294
42: -26.5017319, 13.2108507, -26.4984131, 13.2231607, -38.2113266, 38.1929398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 547

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1722

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5338173, upper bound: 27.5188281
time: 64.54 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5338173, upper bound: 27.5291147
time: 58.48 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -47.0622330, 16.0613918, -46.9438248, 15.9961147, -63.0583496, 63.0052185
1: -25.2315426, 18.7131996, -25.1671581, 18.6372337, -43.6449585, 43.6607513
2: -21.5069637, 18.0249634, -21.4538994, 17.9019051, -39.3740768, 39.4565163
3: -25.0192833, 17.6768932, -24.9816589, 17.4497833, -42.4690666, 42.6585541
4: -28.3558521, 20.2804108, -28.2994347, 20.1458263, -48.5016785, 48.5798454
5: -23.6366615, 20.0078144, -23.5805550, 19.8038826, -43.4405441, 43.5883713
6: -33.7735977, 18.9074478, -33.6980362, 18.8536835, -51.7955093, 51.8363152
7: -30.2929420, 21.4959488, -30.2096214, 21.3781261, -51.1675110, 51.2166100
8: -32.2423935, 24.0857887, -32.1721916, 23.9512711, -56.1936646, 56.2579803
9: -26.7267456, 17.8092804, -26.6728973, 17.6928520, -44.4195976, 44.4821777
10: -36.7109871, 26.0346813, -36.4862518, 25.9679604, -62.6789474, 62.5209351
11: -27.7000351, 25.1087990, -27.3967876, 25.0746250, -52.7746582, 52.5055847
12: -35.3741379, 22.0527058, -35.1162033, 21.9854813, -55.9318390, 55.7389946
13: -40.0408630, 22.6885185, -39.9961090, 22.4600697, -62.5009308, 62.6846275
14: -56.7368622, 13.5374918, -56.3633041, 13.4752226, -70.2120819, 69.9007950
15: -28.2914925, 19.6460762, -28.2424660, 19.5510368, -47.8425293, 47.8885422
16: -35.0007744, 22.4302826, -34.8450203, 22.3621254, -57.3628998, 57.2753029
17: -52.0922318, 27.0596848, -51.7551384, 26.9601784, -79.0524139, 78.8148193
18: -30.2993317, 26.9106140, -29.9896049, 26.8759689, -57.1753006, 56.9002190
19: -17.8119450, 17.6562042, -17.6685886, 17.6377792, -35.3546638, 35.2289810
20: -20.1131306, 17.4897900, -19.9801311, 17.4177666, -37.5308990, 37.4699211
21: -24.6608715, 22.8359566, -24.4437790, 22.8094540, -47.4703255, 47.2797356
22: -25.6161690, 21.6797447, -25.4711227, 21.6385040, -47.2546730, 47.1489143
23: -18.8495216, 21.6812401, -18.6765671, 21.6485291, -40.4980507, 40.3578072
24: -22.5999565, 22.2512608, -22.4097118, 22.2213707, -44.8213272, 44.6609726
25: -19.5882645, 24.5950966, -19.4387569, 24.5634041, -44.1516685, 44.0338516
26: -34.0706291, 29.5143280, -33.7595863, 29.4693336, -63.5399628, 63.2739143
27: -24.1834068, 22.8466911, -24.0542412, 22.8292599, -47.0126648, 46.9009323
28: -18.7718849, 25.4712906, -18.6926270, 25.4211483, -44.0563965, 44.0307312
29: -25.7269726, 24.3403854, -25.5600719, 24.3126717, -50.0396423, 49.9004593
30: -25.8687592, 25.3994884, -25.6731873, 25.3300228, -51.1987839, 51.0726776
31: -22.0530529, 23.8011589, -21.8590508, 23.7869148, -45.8399658, 45.6602097
32: -36.1151810, 15.0107594, -36.0556564, 14.9392328, -49.5781937, 49.6220360
33: -50.8820992, 21.5921421, -50.7057762, 21.3589287, -69.7807007, 69.8347473
34: -50.5894165, 10.0946274, -50.4651527, 9.9432621, -56.6864281, 56.7048225
35: -42.9544334, 17.0481873, -42.8296127, 16.8655872, -57.4667130, 57.5258408
36: -41.9448547, 18.7069931, -41.9120331, 18.6143456, -60.4735413, 60.5408211
37: -55.8088112, 13.1648617, -55.6016693, 13.0704060, -67.8154068, 67.6971893
38: -52.8904419, 15.9560623, -52.8029251, 15.8222523, -68.7126923, 68.7589874
39: -61.8038750, 18.3216228, -61.6506195, 18.1172409, -79.5180359, 79.5661774
40: -48.4619179, 11.8460369, -48.3557205, 11.7300978, -59.2256775, 59.2256546
41: -35.9234695, 18.7030869, -35.8570633, 18.6282234, -53.3685837, 53.3971252
42: -26.5591278, 13.2401762, -26.4724007, 13.1773434, -38.1645775, 38.2012405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=361, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 547

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1722

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5358212, upper bound: 27.5235339
time: 321.01 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5358212, upper bound: 27.5338171
time: 142.83 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -47.1103745, 16.0672569, -47.0477982, 16.0396442, -63.1500168, 63.1150551
1: -25.2705002, 18.7181530, -25.2406998, 18.6866665, -43.7364388, 43.7304649
2: -21.5310402, 18.0296440, -21.5029068, 17.9368019, -39.4346924, 39.4986076
3: -25.0398045, 17.6827698, -25.0228882, 17.5005608, -42.5403671, 42.7056580
4: -28.3749352, 20.2889366, -28.3390579, 20.1877575, -48.5626907, 48.6279945
5: -23.6651134, 20.0146599, -23.6358356, 19.8668098, -43.5319214, 43.6504974
6: -33.7846107, 18.9215851, -33.7322731, 18.8952904, -51.8953323, 51.8866119
7: -30.3367119, 21.5010529, -30.2949066, 21.4477959, -51.2832642, 51.2900772
8: -32.2825012, 24.0944576, -32.2517242, 24.0285854, -56.3110886, 56.3461838
9: -26.7380753, 17.8165779, -26.7013912, 17.7192841, -44.4573593, 44.5179672
10: -36.7244530, 26.0447674, -36.5354309, 25.9993229, -62.7237778, 62.5802002
11: -27.7330151, 25.1153870, -27.4755497, 25.0990829, -52.8320999, 52.5909348
12: -35.3788872, 22.0742302, -35.1436462, 22.0366039, -55.9904327, 55.7930756
13: -40.0625916, 22.7023239, -40.0432701, 22.5327377, -62.5953293, 62.7455940
14: -56.7908478, 13.5440397, -56.4848900, 13.5284567, -70.3193054, 70.0289307
15: -28.3050995, 19.6690826, -28.2866173, 19.6027222, -47.9078217, 47.9556999
16: -35.0316353, 22.4337578, -34.9171219, 22.3869381, -57.4185715, 57.3508797
17: -52.1409721, 27.0667629, -51.8617516, 27.0400505, -79.1810226, 78.9285126
18: -30.3146057, 26.9211216, -30.0524788, 26.9033489, -57.2179565, 56.9736023
19: -17.8268986, 17.6581078, -17.7020798, 17.6465569, -35.3803673, 35.2560730
20: -20.1418953, 17.4983253, -20.0405827, 17.4825191, -37.6244125, 37.5389099
21: -24.6802082, 22.8356457, -24.4854546, 22.8196526, -47.4998627, 47.3210983
22: -25.6288910, 21.6929054, -25.5204906, 21.6745033, -47.3033943, 47.2099991
23: -18.8680096, 21.6872406, -18.7296295, 21.6665268, -40.5345383, 40.4168701
24: -22.6119804, 22.2759094, -22.4782734, 22.2683964, -44.8803787, 44.7541809
25: -19.5993786, 24.6022491, -19.4787521, 24.5858593, -44.1852379, 44.0810013
26: -34.0699005, 29.5331879, -33.7806892, 29.4940109, -63.5639114, 63.3138771
27: -24.1967220, 22.8548126, -24.1014900, 22.8478260, -47.0445480, 46.9563026
28: -18.7879238, 25.4762421, -18.7375069, 25.4501686, -44.1046143, 44.0763283
29: -25.7432899, 24.3461475, -25.6090469, 24.3359528, -50.0792427, 49.9551926
30: -25.8833141, 25.4453297, -25.7529240, 25.4199162, -51.3032303, 51.1982536
31: -22.0686398, 23.7928085, -21.9031162, 23.7801590, -45.8488007, 45.6959229
32: -36.1196671, 15.0220604, -36.0729485, 14.9717741, -49.6501045, 49.6551514
33: -50.8917999, 21.6687927, -50.8499146, 21.4977760, -69.8824158, 70.0595856
34: -50.6006432, 10.1534681, -50.5824928, 10.0477371, -56.7312698, 56.8881836
35: -42.9636421, 17.0978470, -42.9422455, 16.9582520, -57.5247192, 57.6929703
36: -41.9475021, 18.7198715, -41.9260559, 18.6432343, -60.5095520, 60.5683098
37: -55.8220901, 13.2228613, -55.7256851, 13.1754150, -67.9103851, 67.8809586
38: -52.9231186, 15.9691811, -52.8811836, 15.9150429, -68.8381653, 68.8503647
39: -61.8118973, 18.3601570, -61.7547455, 18.1931438, -79.5816650, 79.7069626
40: -48.4726028, 11.8841534, -48.4252548, 11.8027668, -59.2909470, 59.3361359
41: -35.9277840, 18.7175255, -35.8877945, 18.6590042, -53.4132919, 53.4456978
42: -26.5660152, 13.2523470, -26.4966927, 13.2131281, -38.2708969, 38.2366409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 547

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1722

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5338173, upper bound: 27.5235339
time: 124.00 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5338173, upper bound: 27.5338171
time: 563.44 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 689.72 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 689.72
Output dim: 31, lower bound: -27.5358212, upper bound: 27.5188281
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 689.72
Output dim: 31, lower bound: -27.5358212, upper bound: 27.5291147
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 689.72
Output dim: 31, lower bound: -27.5338173, upper bound: 27.5188281
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 689.72
Output dim: 31, lower bound: -27.5338173, upper bound: 27.5291147
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 689.72
Output dim: 31, lower bound: -27.5358212, upper bound: 27.5235339
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 689.72
Output dim: 31, lower bound: -27.5358212, upper bound: 27.5338171
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 689.72
Output dim: 31, lower bound: -27.5338173, upper bound: 27.5235339
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 689.72
Output dim: 31, lower bound: -27.5338173, upper bound: 27.5338171

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 85.46 + 1831.65 = 1917.12 seconds
