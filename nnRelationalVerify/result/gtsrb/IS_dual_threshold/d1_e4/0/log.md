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
execution time: IAR + RelationalAnalysis = 2.72 + 84.93 = 87.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 31, lower bound: -27.5477993, upper bound: 27.5477993

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 999
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 996
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 995
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 998
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 761

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5470964, upper bound: 27.5353020
time: 65.28 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5470964, upper bound: 27.5353020
time: 64.95 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 130.37 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 130.37
Output dim: 31, lower bound: -27.5470964, upper bound: 27.5353020
IS_A2, status: Status.UNKNOWN, split count: 1, time: 130.37
Output dim: 31, lower bound: -27.5470964, upper bound: 27.5353020

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -47.0008163, 16.0386505, -47.0340576, 16.0442181, -63.0450363, 63.0727081
1: -25.2020721, 18.6842384, -25.2280750, 18.6942902, -43.6766663, 43.6924782
2: -21.4531002, 17.9172077, -21.4803200, 17.9416084, -39.3654861, 39.3684921
3: -24.9877853, 17.4869156, -25.0091972, 17.5057068, -42.4934921, 42.4961128
4: -28.2769012, 20.1610775, -28.3106117, 20.1911564, -48.4680557, 48.4716873
5: -23.5914497, 19.8443813, -23.6169586, 19.8705101, -43.4619598, 43.4613419
6: -33.7299881, 18.8481140, -33.7487106, 18.8746529, -51.8138123, 51.8118286
7: -30.2666035, 21.4471645, -30.2893524, 21.4545364, -51.2263870, 51.2380981
8: -32.1429405, 23.9744301, -32.1944427, 24.0302391, -56.1731796, 56.1688728
9: -26.6742706, 17.7009087, -26.7001762, 17.7227764, -44.3970490, 44.4010849
10: -36.4874840, 25.9859562, -36.5156555, 26.0055885, -62.4930725, 62.5016098
11: -27.4052696, 24.9697666, -27.4778633, 25.0316601, -52.4369278, 52.4476318
12: -35.1257744, 21.9794426, -35.1518898, 22.0143833, -55.7355728, 55.7267189
13: -39.9990845, 22.4963074, -40.0374680, 22.5333366, -62.5324211, 62.5337753
14: -56.4385910, 13.5405664, -56.4707375, 13.5375328, -69.9761200, 70.0113068
15: -28.2342091, 19.5851707, -28.2694702, 19.6067619, -47.8409729, 47.8546410
16: -34.9044647, 22.3360672, -34.9278793, 22.3646202, -57.2690849, 57.2639465
17: -51.8123932, 26.9477425, -51.8640480, 26.9952869, -78.8076782, 78.8117905
18: -30.0044823, 26.8155136, -30.0535889, 26.8609276, -56.8654099, 56.8691025
19: -17.6688442, 17.5795403, -17.7043495, 17.6092911, -35.1800194, 35.1868668
20: -20.0276070, 17.4416008, -20.0424671, 17.4633045, -37.4909134, 37.4840698
21: -24.4372044, 22.7428608, -24.4852295, 22.7839241, -47.2211304, 47.2280884
22: -25.4913139, 21.6111698, -25.5269413, 21.6447296, -47.1360435, 47.1381111
23: -18.6974964, 21.6033745, -18.7312393, 21.6378593, -40.3353577, 40.3346138
24: -22.4536324, 22.2453232, -22.4858112, 22.2703724, -44.7240067, 44.7311325
25: -19.4685497, 24.5640335, -19.4859810, 24.5832863, -44.0518341, 44.0500145
26: -33.7649918, 29.4254303, -33.8094063, 29.4645157, -63.2295074, 63.2348366
27: -24.0802784, 22.8032780, -24.1098328, 22.8302021, -46.9104805, 46.9131088
28: -18.7118816, 25.3755913, -18.7467480, 25.4110374, -43.9898415, 43.9889336
29: -25.5684681, 24.2468796, -25.6153336, 24.2899933, -49.8584595, 49.8622131
30: -25.7065582, 25.3368759, -25.7552834, 25.3816185, -51.0881767, 51.0921593
31: -21.8715019, 23.7518921, -21.9081593, 23.7807064, -45.6522064, 45.6600494
32: -36.0832214, 14.9260378, -36.0981064, 14.9523277, -49.5976830, 49.5855789
33: -50.8256035, 21.4672642, -50.8478317, 21.4900227, -69.8642502, 69.8594055
34: -50.5669899, 9.9861469, -50.5874901, 10.0214472, -56.7419090, 56.7309418
35: -42.9368629, 16.9136963, -42.9514008, 16.9406433, -57.5275002, 57.5159302
36: -41.9376297, 18.6007233, -41.9481583, 18.6229172, -60.4807587, 60.4694595
37: -55.7098122, 13.1378326, -55.7335396, 13.1646938, -67.8086243, 67.8055115
38: -52.8500099, 15.9058781, -52.8752937, 15.9249821, -68.7749939, 68.7811737
39: -61.6987076, 18.1788940, -61.7377434, 18.2056961, -79.5014877, 79.5058823
40: -48.4030991, 11.7872925, -48.4234276, 11.8069382, -59.2376404, 59.2368622
41: -35.8779373, 18.5923405, -35.9045410, 18.6262321, -53.3385620, 53.3313217
42: -26.5105896, 13.1870995, -26.5186214, 13.2084751, -38.1968956, 38.1893768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=153, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 999
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 996
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 995
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1621

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5408803, upper bound: 27.5329360
time: 59.19 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5451261, upper bound: 27.5333369
time: 79.72 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -47.0706787, 16.0509262, -47.0716286, 16.0512161, -63.1218948, 63.1225548
1: -25.2535172, 18.6984177, -25.2541637, 18.6985207, -43.7287865, 43.7328377
2: -21.5143814, 17.9455051, -21.5151901, 17.9456043, -39.4238548, 39.4312935
3: -25.0344353, 17.5097370, -25.0350075, 17.5098782, -42.5443115, 42.5447464
4: -28.3525047, 20.1950569, -28.3534660, 20.1952038, -48.5477066, 48.5485229
5: -23.6474953, 19.8749580, -23.6481571, 19.8751011, -43.5225983, 43.5231171
6: -33.7559166, 18.9083252, -33.7562027, 18.9091263, -51.8873062, 51.8804703
7: -30.3101921, 21.4557476, -30.3109055, 21.4569874, -51.2640076, 51.2688446
8: -32.2621002, 24.0395813, -32.2635231, 24.0398235, -56.3019257, 56.3031044
9: -26.7311401, 17.7268963, -26.7317734, 17.7270947, -44.4582367, 44.4586716
10: -36.5439339, 26.0205231, -36.5446167, 26.0209160, -62.5648499, 62.5651398
11: -27.4914436, 25.1129875, -27.4921246, 25.1145592, -52.6060028, 52.6051102
12: -35.1598740, 22.0552940, -35.1601028, 22.0561314, -55.8115921, 55.8022308
13: -40.0828323, 22.5462952, -40.0837746, 22.5467281, -62.6295624, 62.6300697
14: -56.5016708, 13.5434618, -56.5045776, 13.5437078, -70.0453796, 70.0480423
15: -28.3129501, 19.6132011, -28.3138313, 19.6135368, -47.9264870, 47.9270325
16: -34.9385414, 22.3972893, -34.9389572, 22.3979778, -57.3365173, 57.3362465
17: -51.8776703, 27.0590343, -51.8784561, 27.0604134, -78.9380798, 78.9374924
18: -30.0637455, 26.9203472, -30.0641270, 26.9215546, -56.9852982, 56.9844742
19: -17.7131405, 17.6496239, -17.7135124, 17.6504250, -35.2676239, 35.2657204
20: -20.0510635, 17.4896870, -20.0513992, 17.4906197, -37.5416832, 37.5410843
21: -24.4975739, 22.8378754, -24.4980450, 22.8389378, -47.3365097, 47.3359222
22: -25.5379524, 21.6893272, -25.5384846, 21.6903572, -47.2283096, 47.2269249
23: -18.7385788, 21.6808949, -18.7388382, 21.6817780, -40.4203568, 40.4197311
24: -22.4937038, 22.3028736, -22.4939995, 22.3035316, -44.7972336, 44.7968750
25: -19.4933281, 24.6061287, -19.4936218, 24.6067982, -44.1001282, 44.0997505
26: -33.8203697, 29.5162277, -33.8207932, 29.5172749, -63.3376465, 63.3370209
27: -24.1209602, 22.8636475, -24.1213570, 22.8642998, -46.9852600, 46.9850044
28: -18.7537613, 25.4559402, -18.7539997, 25.4568443, -44.0769463, 44.0751953
29: -25.6273918, 24.3481140, -25.6279716, 24.3493042, -49.9766960, 49.9760857
30: -25.7661438, 25.4381618, -25.7665958, 25.4393024, -51.2054443, 51.2047577
31: -21.9183445, 23.8183308, -21.9186172, 23.8190975, -45.7374420, 45.7369461
32: -36.1048355, 14.9799356, -36.1050797, 14.9805660, -49.6468887, 49.6441574
33: -50.8669701, 21.5109940, -50.8674278, 21.5117760, -69.9204636, 69.9299774
34: -50.5964165, 10.0589581, -50.5966110, 10.0598354, -56.8090363, 56.7804413
35: -42.9589729, 16.9681492, -42.9591675, 16.9688396, -57.5751152, 57.5650139
36: -41.9531860, 18.6475830, -41.9533310, 18.6481857, -60.5223846, 60.5214386
37: -55.7408447, 13.1947556, -55.7411041, 13.1955376, -67.8704758, 67.8608093
38: -52.8980217, 15.9386539, -52.8985519, 15.9390507, -68.8370743, 68.8372040
39: -61.7782440, 18.2115536, -61.7790909, 18.2118740, -79.5852203, 79.5956497
40: -48.4425430, 11.8197422, -48.4430962, 11.8207283, -59.2913361, 59.2918396
41: -35.9130135, 18.6673317, -35.9132996, 18.6682110, -53.4157562, 53.4140549
42: -26.5269108, 13.2314873, -26.5271854, 13.2330093, -38.2511330, 38.2449722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=153, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 996
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 998
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1621

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5447302, upper bound: 27.5413540
time: 65.70 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5451261, upper bound: 27.5455910
time: 61.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 129.59 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 129.59
Output dim: 31, lower bound: -27.5408803, upper bound: 27.5329360
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 129.59
Output dim: 31, lower bound: -27.5451261, upper bound: 27.5333369
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 129.59
Output dim: 31, lower bound: -27.5447302, upper bound: 27.5413540
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 129.59
Output dim: 31, lower bound: -27.5451261, upper bound: 27.5455910

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -46.9817123, 15.9664783, -46.9696426, 15.9106445, -62.8923569, 62.9361191
1: -25.1952133, 18.6361961, -25.1967869, 18.6069107, -43.5816383, 43.6129341
2: -21.4447021, 17.8560944, -21.4325142, 17.8294086, -39.2446175, 39.2599640
3: -24.9793053, 17.4600601, -24.9770393, 17.4572220, -42.4365273, 42.4370995
4: -28.2699203, 20.0938358, -28.2630215, 20.0670528, -48.3369751, 48.3568573
5: -23.5846901, 19.7930450, -23.5800896, 19.7751961, -43.3598862, 43.3731346
6: -33.6416206, 18.8367538, -33.5884171, 18.8115616, -51.6673813, 51.6372414
7: -30.2586346, 21.3976402, -30.2486305, 21.3624687, -51.1266251, 51.1477089
8: -32.1346931, 23.9165382, -32.1565399, 23.9244423, -56.0591354, 56.0730782
9: -26.6642151, 17.6903191, -26.6751175, 17.7077084, -44.3719254, 44.3654366
10: -36.4766502, 25.9276409, -36.4766235, 25.9004803, -62.3771286, 62.4042664
11: -27.3872566, 24.9556904, -27.4334164, 25.0026073, -52.3898621, 52.3891068
12: -35.0649757, 21.9643269, -35.0375519, 21.9545708, -55.6086731, 55.5964241
13: -39.9798050, 22.4777641, -39.9931030, 22.4892883, -62.4690933, 62.4708672
14: -56.4176025, 13.4907398, -56.3914642, 13.4453211, -69.8629227, 69.8822021
15: -28.2235222, 19.5565434, -28.2379208, 19.5544853, -47.7780075, 47.7944641
16: -34.8904076, 22.3049889, -34.8838882, 22.3065166, -57.1969223, 57.1888771
17: -51.7917099, 26.8446484, -51.7778664, 26.8051357, -78.5968475, 78.6225128
18: -29.9943371, 26.7839813, -30.0306168, 26.8001575, -56.7944946, 56.8145981
19: -17.6537590, 17.5647831, -17.6736946, 17.5776672, -35.1340637, 35.1358566
20: -20.0062294, 17.4332809, -20.0002460, 17.4410572, -37.4472885, 37.4335251
21: -24.4219246, 22.7278709, -24.4557686, 22.7511959, -47.1731186, 47.1836395
22: -25.4614410, 21.6035042, -25.4653473, 21.6261482, -47.0875893, 47.0688515
23: -18.6867218, 21.5914745, -18.7116299, 21.6105003, -40.2972221, 40.3031044
24: -22.4280052, 22.2156372, -22.4377918, 22.2119122, -44.6399155, 44.6534271
25: -19.4529934, 24.5368690, -19.4550171, 24.5277786, -43.9807739, 43.9918861
26: -33.7370529, 29.4174843, -33.7522049, 29.4402580, -63.1773109, 63.1696892
27: -24.0146637, 22.7958832, -23.9836121, 22.7971916, -46.8118553, 46.7794952
28: -18.6742630, 25.3685684, -18.6765766, 25.3875465, -43.9221153, 43.9052124
29: -25.5171223, 24.2406578, -25.5117283, 24.2842140, -49.8013382, 49.7523880
30: -25.6726723, 25.3256321, -25.6891365, 25.3604488, -51.0331192, 51.0147705
31: -21.8551598, 23.7201118, -21.8746452, 23.7191963, -45.5743561, 45.5947571
32: -35.9705429, 14.9123850, -35.8889542, 14.8800583, -49.4060440, 49.3612061
33: -50.8062668, 21.4550381, -50.8123817, 21.4473343, -69.7954712, 69.8103943
34: -50.4977150, 9.9793186, -50.4610252, 9.9802990, -56.6275139, 56.5950508
35: -42.8979645, 16.9093742, -42.8790665, 16.9160767, -57.4625854, 57.4380722
36: -41.8476448, 18.5960732, -41.7821960, 18.5737038, -60.3378983, 60.2990952
37: -55.6581841, 13.1270485, -55.6395226, 13.1200361, -67.7081146, 67.6998749
38: -52.7800751, 15.8973923, -52.7482719, 15.8926964, -68.6727753, 68.6456604
39: -61.6541939, 18.1709557, -61.6549988, 18.1746235, -79.4259186, 79.4158249
40: -48.3105354, 11.7783537, -48.2528915, 11.7512932, -59.0872345, 59.0563049
41: -35.7978210, 18.5827217, -35.7544899, 18.5718346, -53.1949844, 53.1699791
42: -26.4456387, 13.1767588, -26.3961201, 13.1540070, -38.0819473, 38.0510635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=361, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 996
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 998
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5305256, upper bound: 27.5274252
time: 62.97 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5354428, upper bound: 27.5274252
time: 62.76 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -46.9998894, 16.0369720, -47.0318680, 16.0402756, -63.0401649, 63.0688400
1: -25.2017155, 18.6830482, -25.2272682, 18.6914864, -43.6710663, 43.6904640
2: -21.4527321, 17.9159603, -21.4794273, 17.9386654, -39.3566132, 39.3663406
3: -24.9875431, 17.4860992, -25.0086021, 17.5038166, -42.4913597, 42.4947014
4: -28.2766094, 20.1595802, -28.3099327, 20.1876335, -48.4642410, 48.4695129
5: -23.5911484, 19.8432446, -23.6162434, 19.8678703, -43.4590187, 43.4594879
6: -33.7287064, 18.8477669, -33.7456894, 18.8738441, -51.8114243, 51.7966537
7: -30.2662354, 21.4460869, -30.2885113, 21.4520226, -51.2172089, 51.2361336
8: -32.1426392, 23.9729881, -32.1937561, 24.0268478, -56.1694870, 56.1667442
9: -26.6734447, 17.7003269, -26.6981926, 17.7213821, -44.3948288, 44.3985214
10: -36.4870338, 25.9843311, -36.5145721, 26.0016785, -62.4887123, 62.4989014
11: -27.4038658, 24.9692936, -27.4747009, 25.0305195, -52.4343872, 52.4439926
12: -35.1247711, 21.9787502, -35.1496964, 22.0128727, -55.7331009, 55.7113304
13: -39.9984283, 22.4952469, -40.0360031, 22.5308800, -62.5293083, 62.5312500
14: -56.4374695, 13.5399160, -56.4681129, 13.5363560, -69.9738235, 70.0080261
15: -28.2338524, 19.5838242, -28.2685833, 19.6035633, -47.8374176, 47.8524094
16: -34.9036407, 22.3351364, -34.9259644, 22.3624001, -57.2660408, 57.2611008
17: -51.8115120, 26.9453888, -51.8619804, 26.9897652, -78.8012772, 78.8073730
18: -30.0038528, 26.8129425, -30.0521317, 26.8549156, -56.8587685, 56.8650742
19: -17.6682339, 17.5784187, -17.7029190, 17.6066360, -35.1737061, 35.1841469
20: -20.0267220, 17.4412193, -20.0403843, 17.4624481, -37.4891701, 37.4816055
21: -24.4364128, 22.7417336, -24.4833908, 22.7812614, -47.2176743, 47.2251244
22: -25.4904919, 21.6108284, -25.5250206, 21.6439457, -47.1344376, 47.1358490
23: -18.6969299, 21.6024017, -18.7299385, 21.6355476, -40.3324776, 40.3323402
24: -22.4527473, 22.2427082, -22.4837513, 22.2646561, -44.7174034, 44.7264595
25: -19.4679585, 24.5624027, -19.4845905, 24.5794678, -44.0474243, 44.0469933
26: -33.7641296, 29.4248581, -33.8073654, 29.4631824, -63.2273102, 63.2322235
27: -24.0787239, 22.8029900, -24.1062050, 22.8295403, -46.9082642, 46.9091949
28: -18.7102680, 25.3752747, -18.7431240, 25.4102936, -43.9915009, 43.9852753
29: -25.5663490, 24.2465439, -25.6105137, 24.2892075, -49.8555565, 49.8570557
30: -25.7048264, 25.3364277, -25.7512589, 25.3805542, -51.0853806, 51.0876846
31: -21.8706284, 23.7496967, -21.9061050, 23.7754898, -45.6461182, 45.6557999
32: -36.0817986, 14.9255342, -36.0947647, 14.9511595, -49.5950050, 49.5717659
33: -50.8243332, 21.4668579, -50.8449478, 21.4890480, -69.8617096, 69.8436813
34: -50.5654221, 9.9857578, -50.5837860, 10.0205317, -56.7393608, 56.7015991
35: -42.9354515, 16.9135132, -42.9483528, 16.9402695, -57.5257874, 57.4985275
36: -41.9357338, 18.6004601, -41.9441299, 18.6222801, -60.4782257, 60.4626160
37: -55.7081146, 13.1374273, -55.7295609, 13.1637754, -67.8059540, 67.7929077
38: -52.8482246, 15.9053555, -52.8711090, 15.9238033, -68.7720261, 68.7764664
39: -61.6969719, 18.1785202, -61.7337570, 18.2049274, -79.4988861, 79.4962769
40: -48.4010620, 11.7868290, -48.4186325, 11.8058119, -59.2344818, 59.2235527
41: -35.8762856, 18.5918694, -35.9012604, 18.6251297, -53.3359375, 53.3192406
42: -26.5093250, 13.1865082, -26.5156364, 13.2070847, -38.1940155, 38.1707611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 996
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 998
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 581

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5305256, upper bound: 27.5276098
time: 57.09 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5394011, upper bound: 27.5276098
time: 57.48 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -47.0062714, 15.9173737, -47.0525169, 15.9790688, -62.9853401, 62.9698906
1: -25.2222519, 18.6110306, -25.2473221, 18.6504917, -43.6492157, 43.6377831
2: -21.4666023, 17.8333073, -21.5068073, 17.8844604, -39.3153076, 39.3104172
3: -25.0022736, 17.4612389, -25.0265312, 17.4830265, -42.4852982, 42.4877701
4: -28.3049202, 20.0709705, -28.3465042, 20.1279640, -48.4328842, 48.4174728
5: -23.6105957, 19.7796116, -23.6414108, 19.8237534, -43.4343491, 43.4210205
6: -33.5956230, 18.8452530, -33.6678352, 18.8977642, -51.7127457, 51.7340927
7: -30.2694702, 21.3636818, -30.3029137, 21.4074459, -51.1736374, 51.1691284
8: -32.2241974, 23.9337997, -32.2553101, 23.9819221, -56.2061195, 56.1891098
9: -26.7061081, 17.7118301, -26.7217426, 17.7164841, -44.4225922, 44.4335709
10: -36.5048752, 25.9154377, -36.5337753, 25.9626083, -62.4674835, 62.4492111
11: -27.4469681, 25.0839653, -27.4741001, 25.1005058, -52.5474739, 52.5580673
12: -35.0455627, 21.9954910, -35.0993652, 22.0410652, -55.6813049, 55.6753387
13: -40.0384903, 22.5022316, -40.0645103, 22.5281639, -62.5666542, 62.5667419
14: -56.4224014, 13.4512320, -56.4836311, 13.4939137, -69.9163132, 69.9348602
15: -28.2814217, 19.5609474, -28.3031616, 19.5848999, -47.8663216, 47.8641090
16: -34.8945122, 22.3391800, -34.9249191, 22.3668823, -57.2613945, 57.2640991
17: -51.7915115, 26.8688889, -51.8577995, 26.9573860, -78.7489014, 78.7266846
18: -30.0407734, 26.8596287, -30.0539722, 26.8900375, -56.9308090, 56.9136009
19: -17.6824932, 17.6180000, -17.6984119, 17.6356792, -35.2166214, 35.2197418
20: -20.0088329, 17.4674339, -20.0300102, 17.4823112, -37.4911423, 37.4974442
21: -24.4680843, 22.8051300, -24.4827595, 22.8239059, -47.2919922, 47.2878876
22: -25.4763470, 21.6707802, -25.5086174, 21.6827068, -47.1590538, 47.1774864
23: -18.7189713, 21.6535378, -18.7280731, 21.6699066, -40.3888779, 40.3816109
24: -22.4456940, 22.2444115, -22.4683819, 22.2738533, -44.7195473, 44.7127914
25: -19.4623337, 24.5506535, -19.4780846, 24.5796585, -44.0419922, 44.0287399
26: -33.7631836, 29.4919434, -33.7928467, 29.5093098, -63.2724915, 63.2847900
27: -23.9947300, 22.8306217, -24.0557384, 22.8569336, -46.8516617, 46.8863602
28: -18.6835785, 25.4324417, -18.7163639, 25.4498100, -43.9932289, 44.0074387
29: -25.5237656, 24.3423386, -25.5766163, 24.3430824, -49.8668480, 49.9189529
30: -25.6999664, 25.4169960, -25.7326546, 25.4280643, -51.1280289, 51.1496506
31: -21.8848267, 23.7568359, -21.9022865, 23.7873211, -45.6721497, 45.6591225
32: -35.8957253, 14.9077148, -35.9924355, 14.9669561, -49.4225502, 49.4525757
33: -50.8315201, 21.4683132, -50.8481331, 21.4995804, -69.8714600, 69.8611450
34: -50.4699402, 10.0177870, -50.5273514, 10.0529766, -56.6730843, 56.6660156
35: -42.8866501, 16.9435902, -42.9202423, 16.9645424, -57.4972610, 57.5000839
36: -41.7872124, 18.5983734, -41.8633804, 18.6435471, -60.3519897, 60.3785324
37: -55.6468124, 13.1500387, -55.6895142, 13.1847181, -67.7648773, 67.7603226
38: -52.7710495, 15.9063568, -52.8286819, 15.9305458, -68.7015991, 68.7350388
39: -61.6954498, 18.1804638, -61.7346191, 18.2039356, -79.4951782, 79.5200806
40: -48.2720108, 11.7641077, -48.3505630, 11.8117762, -59.1107941, 59.1414108
41: -35.7629700, 18.6129112, -35.8332138, 18.6586113, -53.2544098, 53.2704887
42: -26.4044228, 13.1770372, -26.4622440, 13.2226677, -38.1128540, 38.1300583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=151, inp2_unstable=153, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=361, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 996
type: B, layer: 1, pos: 996
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 998
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5392148, upper bound: 27.5309762
time: 64.89 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5392148, upper bound: 27.5358639
time: 63.14 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -47.0684662, 16.0469971, -47.0706863, 16.0495377, -63.1180038, 63.1176834
1: -25.2527122, 18.6955986, -25.2538185, 18.6973171, -43.7267876, 43.7272491
2: -21.5134811, 17.9425831, -21.5148163, 17.9443531, -39.4216652, 39.4224281
3: -25.0338440, 17.5078354, -25.0347519, 17.5090561, -42.5429001, 42.5425873
4: -28.3518257, 20.1915665, -28.3531723, 20.1937237, -48.5455475, 48.5447388
5: -23.6467743, 19.8723106, -23.6478653, 19.8739700, -43.5207443, 43.5201759
6: -33.7528839, 18.9075050, -33.7549210, 18.9087772, -51.8721619, 51.8781204
7: -30.3093414, 21.4532223, -30.3105316, 21.4559021, -51.2620316, 51.2596741
8: -32.2614326, 24.0361938, -32.2632294, 24.0383682, -56.2998009, 56.2994232
9: -26.7291794, 17.7255116, -26.7309418, 17.7265034, -44.4556808, 44.4564514
10: -36.5428467, 26.0166206, -36.5441475, 26.0192490, -62.5620956, 62.5607681
11: -27.4883041, 25.1118298, -27.4907055, 25.1140709, -52.6023750, 52.6025352
12: -35.1576958, 22.0537815, -35.1591339, 22.0554733, -55.7961807, 55.7997894
13: -40.0813637, 22.5438175, -40.0831375, 22.5456676, -62.6270294, 62.6269531
14: -56.4990311, 13.5422592, -56.5034714, 13.5430689, -70.0420990, 70.0457306
15: -28.3120766, 19.6100178, -28.3134727, 19.6121902, -47.9242668, 47.9234924
16: -34.9366112, 22.3950615, -34.9381294, 22.3970165, -57.3336258, 57.3331909
17: -51.8755951, 27.0535221, -51.8775787, 27.0580826, -78.9336777, 78.9310989
18: -30.0622902, 26.9143219, -30.0634937, 26.9189949, -56.9812851, 56.9778137
19: -17.7117119, 17.6469765, -17.7129021, 17.6493053, -35.2649040, 35.2593842
20: -20.0489807, 17.4888134, -20.0504971, 17.4902306, -37.5392113, 37.5393105
21: -24.4957390, 22.8351974, -24.4972668, 22.8377991, -47.3335381, 47.3324661
22: -25.5360355, 21.6885395, -25.5376740, 21.6900215, -47.2260590, 47.2249069
23: -18.7372818, 21.6785583, -18.7382832, 21.6807823, -40.4180641, 40.4168396
24: -22.4916706, 22.2971401, -22.4931126, 22.3008976, -44.7925682, 44.7902527
25: -19.4919548, 24.6023178, -19.4930172, 24.6051769, -44.0971298, 44.0953369
26: -33.8183441, 29.5148659, -33.8199120, 29.5166740, -63.3350182, 63.3347778
27: -24.1173134, 22.8629742, -24.1198082, 22.8640289, -46.9813423, 46.9827805
28: -18.7501106, 25.4551811, -18.7523823, 25.4565239, -44.0732727, 44.0768471
29: -25.6226025, 24.3473415, -25.6258602, 24.3489876, -49.9715881, 49.9732018
30: -25.7620964, 25.4371071, -25.7648430, 25.4388351, -51.2009315, 51.2019501
31: -21.9162941, 23.8131046, -21.9177532, 23.8168678, -45.7331619, 45.7308578
32: -36.1015167, 14.9787674, -36.1036568, 14.9800272, -49.6330719, 49.6414795
33: -50.8640938, 21.5100460, -50.8661880, 21.5113831, -69.9047852, 69.9274292
34: -50.5926971, 10.0580263, -50.5950432, 10.0594320, -56.7796783, 56.7778969
35: -42.9559326, 16.9678020, -42.9577637, 16.9687119, -57.5577087, 57.5632858
36: -41.9491425, 18.6469364, -41.9514503, 18.6479187, -60.5155029, 60.5188980
37: -55.7368622, 13.1938038, -55.7393723, 13.1951256, -67.8579025, 67.8581696
38: -52.8938370, 15.9374237, -52.8967552, 15.9385300, -68.8323669, 68.8341827
39: -61.7742119, 18.2107182, -61.7773705, 18.2115040, -79.5756454, 79.5930099
40: -48.4377441, 11.8186102, -48.4410591, 11.8202553, -59.2779922, 59.2886353
41: -35.9097290, 18.6662273, -35.9116669, 18.6677132, -53.4036407, 53.4114838
42: -26.5239258, 13.2300797, -26.5259190, 13.2323952, -38.2325363, 38.2420731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=151, inp2_unstable=153, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 999
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 996
type: B, layer: 1, pos: 996
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 998
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 1668

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5392148, upper bound: 27.5351244
time: 61.34 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5394012, upper bound: 27.5398089
time: 87.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 151.35 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 151.35
Output dim: 31, lower bound: -27.5305256, upper bound: 27.5274252
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 151.35
Output dim: 31, lower bound: -27.5354428, upper bound: 27.5274252
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 151.35
Output dim: 31, lower bound: -27.5305256, upper bound: 27.5276098
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 151.35
Output dim: 31, lower bound: -27.5394011, upper bound: 27.5276098
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 151.35
Output dim: 31, lower bound: -27.5392148, upper bound: 27.5309762
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 151.35
Output dim: 31, lower bound: -27.5392148, upper bound: 27.5358639
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 151.35
Output dim: 31, lower bound: -27.5392148, upper bound: 27.5351244
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 151.35
Output dim: 31, lower bound: -27.5394012, upper bound: 27.5398089

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -46.9736786, 15.9639816, -46.9426498, 15.9021950, -62.8758736, 62.9066315
1: -25.1913261, 18.6325397, -25.1837807, 18.5949535, -43.5658112, 43.5948181
2: -21.4370461, 17.8543549, -21.4073582, 17.8236504, -39.2303734, 39.2304077
3: -24.9676704, 17.4567490, -24.9377079, 17.4461174, -42.4137878, 42.3944550
4: -28.2623138, 20.0918407, -28.2376556, 20.0604057, -48.3227196, 48.3294983
5: -23.5740509, 19.7900620, -23.5441971, 19.7652225, -43.3392715, 43.3342590
6: -33.6385117, 18.8342209, -33.5780258, 18.8031063, -51.6553955, 51.6234360
7: -30.2486954, 21.3952389, -30.2151680, 21.3543777, -51.1078110, 51.1093483
8: -32.1256218, 23.9138012, -32.1259842, 23.9154034, -56.0410233, 56.0397873
9: -26.6575317, 17.6873226, -26.6528740, 17.6976700, -44.3552017, 44.3401947
10: -36.4723358, 25.9172344, -36.4621964, 25.8654480, -62.3377838, 62.3794327
11: -27.3835030, 24.9415550, -27.4209709, 24.9547901, -52.3382950, 52.3625259
12: -35.0611534, 21.9431572, -35.0246239, 21.8833504, -55.5311050, 55.5615501
13: -39.9672012, 22.4733181, -39.9517059, 22.4743767, -62.4415779, 62.4250259
14: -56.4101219, 13.4731703, -56.3661575, 13.3858604, -69.7959824, 69.8393250
15: -28.2188263, 19.5531597, -28.2223949, 19.5432091, -47.7620354, 47.7755547
16: -34.8846893, 22.3025627, -34.8647537, 22.2983589, -57.1830482, 57.1673164
17: -51.7868347, 26.8209820, -51.7614059, 26.7251015, -78.5119324, 78.5823898
18: -29.9903984, 26.7650642, -30.0174656, 26.7372208, -56.7276192, 56.7825317
19: -17.6501560, 17.5609360, -17.6616783, 17.5646553, -35.1159439, 35.1197891
20: -20.0025520, 17.4266243, -19.9878616, 17.4187279, -37.4212799, 37.4144859
21: -24.4178429, 22.7167664, -24.4420414, 22.7136097, -47.1314545, 47.1588058
22: -25.4579029, 21.5928268, -25.4534626, 21.5904408, -47.0483437, 47.0459900
23: -18.6838531, 21.5847607, -18.7020798, 21.5881767, -40.2720299, 40.2868423
24: -22.4240131, 22.2073288, -22.4246140, 22.1839275, -44.6079407, 44.6319427
25: -19.4494267, 24.5274830, -19.4432049, 24.4960518, -43.9454803, 43.9706879
26: -33.7328415, 29.3943062, -33.7381744, 29.3619614, -63.0948029, 63.1324806
27: -24.0104008, 22.7889252, -23.9692574, 22.7739239, -46.7843246, 46.7581825
28: -18.6703415, 25.3665199, -18.6633949, 25.3806953, -43.9097061, 43.8891830
29: -25.5135422, 24.2286949, -25.4997025, 24.2438545, -49.7573967, 49.7283974
30: -25.6692848, 25.3147507, -25.6779060, 25.3237114, -50.9929962, 50.9926567
31: -21.8498955, 23.7141113, -21.8570004, 23.6996288, -45.5495224, 45.5711136
32: -35.9644890, 14.9091015, -35.8686218, 14.8691778, -49.3878326, 49.3358040
33: -50.7930069, 21.4514618, -50.7688065, 21.4353027, -69.7696228, 69.7608566
34: -50.4896164, 9.9756699, -50.4337044, 9.9680920, -56.6066284, 56.5657158
35: -42.8869553, 16.9067993, -42.8419189, 16.9074478, -57.4423447, 57.3983688
36: -41.8415985, 18.5934677, -41.7616959, 18.5649033, -60.3224106, 60.2749557
37: -55.6530762, 13.1227255, -55.6224823, 13.1054430, -67.6865845, 67.6768036
38: -52.7740097, 15.8911552, -52.7278595, 15.8721943, -68.6462021, 68.6190186
39: -61.6436577, 18.1687794, -61.6196404, 18.1672516, -79.4069214, 79.3764648
40: -48.3022995, 11.7764101, -48.2251587, 11.7447977, -59.0719299, 59.0256271
41: -35.7931137, 18.5798378, -35.7386665, 18.5620918, -53.1796951, 53.1497574
42: -26.4428749, 13.1705561, -26.3868923, 13.1333542, -38.0579910, 38.0362473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=361, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 996
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 998
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 679

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5293098, upper bound: 27.5261226
time: 46.30 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5293098, upper bound: 27.5262153
time: 574.43 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -46.9674683, 15.9581261, -47.0138893, 15.9285259, -62.8959961, 62.9720154
1: -25.1874981, 18.6273327, -25.2163353, 18.6283340, -43.5961990, 43.6226425
2: -21.4360161, 17.8516598, -21.4501495, 17.9161873, -39.3322105, 39.2697182
3: -24.9702301, 17.4540558, -24.9832993, 17.6321487, -42.6023788, 42.4373550
4: -28.2594624, 20.0897789, -28.2863541, 20.1629372, -48.4224014, 48.3761330
5: -23.5763817, 19.7874184, -23.5990715, 19.9164276, -43.4928093, 43.3864899
6: -33.6355133, 18.8265247, -33.6279526, 18.8258400, -51.6928329, 51.6642303
7: -30.2483673, 21.3910065, -30.2775726, 21.4075775, -51.1689453, 51.1637421
8: -32.1271324, 23.9099121, -32.1774216, 23.9819717, -56.1091042, 56.0873337
9: -26.6452293, 17.6855183, -26.6883583, 17.7989769, -44.4442062, 44.3738785
10: -36.4705582, 25.9117813, -36.6580391, 25.9276695, -62.3982277, 62.5698204
11: -27.3775520, 24.9453888, -27.6785717, 25.0058632, -52.3834152, 52.6239624
12: -35.0583649, 21.9488411, -35.2624588, 21.9744072, -55.6142044, 55.8090858
13: -39.9471703, 22.4698792, -39.9760780, 22.6490784, -62.5962486, 62.4459572
14: -56.4067039, 13.4781580, -56.6818008, 13.4472380, -69.8539429, 70.1599579
15: -28.2012501, 19.5508347, -28.2317905, 19.6130333, -47.8142853, 47.7826233
16: -34.8806763, 22.2989311, -34.9835815, 22.3450699, -57.2257462, 57.2825127
17: -51.7835541, 26.8260269, -52.0453377, 26.8118896, -78.5954437, 78.8713684
18: -29.9871273, 26.7690811, -30.2838249, 26.8014755, -56.7886047, 57.0529060
19: -17.6476498, 17.5614967, -17.7901630, 17.5853996, -35.1340103, 35.2556915
20: -20.0011559, 17.4271507, -20.0942936, 17.4493313, -37.4504852, 37.5214462
21: -24.4159107, 22.7197800, -24.6418610, 22.7542744, -47.1701851, 47.3616409
22: -25.4522934, 21.5930328, -25.5609455, 21.6314564, -47.0802307, 47.1539764
23: -18.6815243, 21.5837402, -18.8431168, 21.6202450, -40.3017693, 40.4268570
24: -22.4174519, 22.1939125, -22.5588989, 22.1935806, -44.6110306, 44.7528114
25: -19.4438438, 24.5281086, -19.5642719, 24.5307808, -43.9746246, 44.0923805
26: -33.7273178, 29.4001083, -34.0198364, 29.4592209, -63.1865387, 63.4199448
27: -24.0065289, 22.7813663, -24.0661526, 22.7885380, -46.7950668, 46.8475189
28: -18.6658058, 25.3647366, -18.7154255, 25.4082527, -43.9320335, 43.9410782
29: -25.5070057, 24.2285290, -25.6322041, 24.2812271, -49.7882309, 49.8607330
30: -25.6638985, 25.3139687, -25.8091450, 25.3705444, -51.0344429, 51.1231155
31: -21.8473892, 23.7058601, -22.0293407, 23.7080650, -45.5554543, 45.7351990
32: -35.9598808, 14.9056330, -35.9166641, 14.9225464, -49.4399567, 49.3816299
33: -50.7943878, 21.4490891, -50.8400192, 21.6089172, -69.9430389, 69.8229370
34: -50.4893265, 9.9742842, -50.4685173, 10.0778379, -56.7129517, 56.5887680
35: -42.8868713, 16.9057217, -42.8870316, 17.0503426, -57.5856094, 57.4386787
36: -41.8321152, 18.5932808, -41.7834435, 18.6464195, -60.3970184, 60.2959518
37: -55.6475449, 13.1129847, -55.7233696, 13.1504917, -67.7265930, 67.7678299
38: -52.7697754, 15.8770084, -52.7769508, 15.9248161, -68.6945953, 68.6539612
39: -61.6358299, 18.1671448, -61.6911201, 18.3336372, -79.5631104, 79.4474411
40: -48.2960968, 11.7717838, -48.2843208, 11.8212004, -59.1415787, 59.0779266
41: -35.7900543, 18.5777454, -35.7795448, 18.6233292, -53.2469559, 53.1870537
42: -26.4411469, 13.1605148, -26.4513016, 13.1748028, -38.1016312, 38.0959473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=361, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 999
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 996
type: A, layer: 1, pos: 996
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 995
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1671

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 679

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5293098, upper bound: 27.5261226
time: 66.00 seconds

## Relational analysis of IS_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5293098, upper bound: 27.5262153
time: 61.81 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -46.9918633, 16.0344429, -47.0048828, 16.0318375, -63.0237007, 63.0393257
1: -25.1978130, 18.6793728, -25.2142773, 18.6795387, -43.6552467, 43.6723633
2: -21.4450359, 17.9142342, -21.4542427, 17.9329166, -39.3423882, 39.3368034
3: -24.9758949, 17.4827614, -24.9692707, 17.4927464, -42.4686432, 42.4520340
4: -28.2689800, 20.1575851, -28.2845745, 20.1809750, -48.4499550, 48.4421616
5: -23.5805264, 19.8402691, -23.5803833, 19.8579025, -43.4384308, 43.4206543
6: -33.7255859, 18.8452396, -33.7352715, 18.8653812, -51.7994461, 51.7828636
7: -30.2563133, 21.4436913, -30.2550278, 21.4439259, -51.1983490, 51.1977692
8: -32.1335907, 23.9702549, -32.1632462, 24.0177612, -56.1513519, 56.1334991
9: -26.6667938, 17.6973534, -26.6759701, 17.7113361, -44.3781281, 44.3733215
10: -36.4827156, 25.9739151, -36.5001564, 25.9666328, -62.4493484, 62.4740715
11: -27.4001236, 24.9551582, -27.4622879, 24.9826870, -52.3828125, 52.4174461
12: -35.1209259, 21.9576244, -35.1367493, 21.9416580, -55.6555634, 55.6764450
13: -39.9858322, 22.4908142, -39.9946442, 22.5159206, -62.5017548, 62.4854584
14: -56.4299736, 13.5223484, -56.4428024, 13.4768448, -69.9068146, 69.9651489
15: -28.2291336, 19.5804501, -28.2530231, 19.5922890, -47.8214226, 47.8334732
16: -34.8978920, 22.3327293, -34.9068565, 22.3542290, -57.2521210, 57.2395859
17: -51.8066216, 26.9217072, -51.8455696, 26.9097404, -78.7163620, 78.7672729
18: -29.9998989, 26.7940273, -30.0389519, 26.7919350, -56.7918320, 56.8329773
19: -17.6646309, 17.5745621, -17.6909065, 17.5936260, -35.1556053, 35.1680679
20: -20.0230408, 17.4345875, -20.0279961, 17.4401035, -37.4631424, 37.4625854
21: -24.4323349, 22.7306252, -24.4696941, 22.7436790, -47.1760139, 47.2003174
22: -25.4869766, 21.6001492, -25.5131550, 21.6082191, -47.0951958, 47.1133041
23: -18.6940784, 21.5956726, -18.7204018, 21.6132450, -40.3073235, 40.3160744
24: -22.4487572, 22.2343864, -22.4705696, 22.2366753, -44.6854324, 44.7049561
25: -19.4643879, 24.5530071, -19.4727707, 24.5477467, -44.0121346, 44.0257797
26: -33.7599106, 29.4016628, -33.7933235, 29.3848820, -63.1447906, 63.1949844
27: -24.0744705, 22.7960358, -24.0918579, 22.8062820, -46.8807526, 46.8878937
28: -18.7063332, 25.3732414, -18.7299080, 25.4034233, -43.9791069, 43.9692078
29: -25.5627766, 24.2346268, -25.5984936, 24.2488670, -49.8116455, 49.8331223
30: -25.7014313, 25.3255539, -25.7399998, 25.3438148, -51.0452461, 51.0655518
31: -21.8653641, 23.7436829, -21.8884525, 23.7559357, -45.6212997, 45.6321335
32: -36.0757256, 14.9222336, -36.0744591, 14.9402781, -49.5767937, 49.5463562
33: -50.8110504, 21.4632816, -50.8013535, 21.4769859, -69.8359299, 69.7941284
34: -50.5572968, 9.9821091, -50.5564461, 10.0083551, -56.7185135, 56.6722755
35: -42.9244308, 16.9109535, -42.9111710, 16.9316330, -57.5055466, 57.4588356
36: -41.9296646, 18.5978184, -41.9236412, 18.6135178, -60.4626999, 60.4384995
37: -55.7029686, 13.1331272, -55.7125397, 13.1492128, -67.7843323, 67.7698669
38: -52.8421593, 15.8991299, -52.8506813, 15.9033127, -68.7454681, 68.7498093
39: -61.6863976, 18.1763611, -61.6983490, 18.1975403, -79.4798584, 79.4569016
40: -48.3928146, 11.7848883, -48.3908920, 11.7992830, -59.2191849, 59.1928406
41: -35.8715591, 18.5889606, -35.8854065, 18.6154041, -53.3206711, 53.2989998
42: -26.5065613, 13.1803017, -26.5064068, 13.1864367, -38.1700668, 38.1559296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 996
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1004
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1004
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 998
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1668

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 679

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5293098, upper bound: 27.5263031
time: 132.81 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5293098, upper bound: 27.5263964
time: 59.04 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 194.24 seconds
IS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 194.24
Output dim: 31, lower bound: -27.5293098, upper bound: 27.5261226
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 194.24
Output dim: 31, lower bound: -27.5293098, upper bound: 27.5262153
IS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 194.24
Output dim: 31, lower bound: -27.5293098, upper bound: 27.5261226
IS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 194.24
Output dim: 31, lower bound: -27.5293098, upper bound: 27.5262153
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 194.24
Output dim: 31, lower bound: -27.5293098, upper bound: 27.5263031
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 194.24
Output dim: 31, lower bound: -27.5293098, upper bound: 27.5263964
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 194.24
Output dim: 31, lower bound: -27.5394011, upper bound: 27.5276098
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 194.24
Output dim: 31, lower bound: -27.5392148, upper bound: 27.5309762
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 194.24
Output dim: 31, lower bound: -27.5392148, upper bound: 27.5358639
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 194.24
Output dim: 31, lower bound: -27.5392148, upper bound: 27.5351244
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 194.24
Output dim: 31, lower bound: -27.5394012, upper bound: 27.5398089

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 87.65 + 1875.31 = 1962.96 seconds
