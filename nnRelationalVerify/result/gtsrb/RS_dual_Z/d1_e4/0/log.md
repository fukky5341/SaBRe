## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 0)
Time budget: 1800 seconds
Split limit: 100


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.71 + 85.12 = 87.82 seconds
status: Status.UNKNOWN
relational distance
Output dim: 31, lower bound: -27.5477993, upper bound: 27.5477993

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1539

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5419843, upper bound: 27.5373015
time: 66.54 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5373015, upper bound: 27.5419843
time: 67.17 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 133.83 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 133.83
Output dim: 31, lower bound: -27.5419843, upper bound: 27.5373015
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 133.83
Output dim: 31, lower bound: -27.5373015, upper bound: 27.5419843

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7332382, 43.7331047
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4324341, 39.4322891
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8835068, 51.8830605
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2753830, 51.2751617
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.8130493, 55.8131027
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2682915, 35.2682953
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0781746, 44.0782623
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6481323, 49.6481171
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9204559, 69.9208984
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.8090515, 56.8097534
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5761986, 57.5767365
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5228577, 60.5228195
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8712006, 67.8713913
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5877228, 79.5879517
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2939072, 59.2938919
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4171906, 53.4171982
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2465515, 38.2459412

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1539

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 721

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5309571, upper bound: 27.5366181
time: 114.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5413003, upper bound: 27.5262839
time: 65.42 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7331009, 43.7332344
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4322891, 39.4324379
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8830643, 51.8834991
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2751617, 51.2753792
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.8131104, 55.8130608
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2682915, 35.2682877
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0782661, 44.0781746
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6481171, 49.6481285
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9208984, 69.9204559
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.8097534, 56.8090439
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5767326, 57.5761986
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5228271, 60.5228577
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8713837, 67.8712082
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5879364, 79.5877304
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2938919, 59.2938995
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4171982, 53.4171906
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2459412, 38.2465439

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1539

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 721

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5262839, upper bound: 27.5413003
time: 56.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5366181, upper bound: 27.5309571
time: 57.57 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 116.87 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 116.87
Output dim: 31, lower bound: -27.5309571, upper bound: 27.5366181
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 116.87
Output dim: 31, lower bound: -27.5413003, upper bound: 27.5262839
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 116.87
Output dim: 31, lower bound: -27.5262839, upper bound: 27.5413003
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 116.87
Output dim: 31, lower bound: -27.5366181, upper bound: 27.5309571

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7316093, 43.7312698
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4292717, 39.4287262
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8802490, 51.8801422
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2716141, 51.2709198
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.8066711, 55.8074150
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2702980, 35.2699623
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0784302, 44.0783577
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6438408, 49.6442795
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9100571, 69.9116669
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.7929611, 56.7954903
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5658340, 57.5675278
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5219040, 60.5219727
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8658905, 67.8666611
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5838470, 79.5845184
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2898254, 59.2902679
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4137955, 53.4141617
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2453918, 38.2452431

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1539

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 627

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5300723, upper bound: 27.5237113
time: 61.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5179921, upper bound: 27.5357339
time: 58.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7313957, 43.7314796
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4288750, 39.4291267
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8805847, 51.8798141
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2711411, 51.2713928
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.8073730, 55.8067169
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2699623, 35.2702980
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0782623, 44.0785179
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6442986, 49.6438217
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9112167, 69.9104919
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.7947845, 56.7936707
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5669937, 57.5663719
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5220108, 60.5218773
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8664703, 67.8660660
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5842896, 79.5840759
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2902832, 59.2898293
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4141541, 53.4138031
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2458496, 38.2447853

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1539

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 627

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5404152, upper bound: 27.5133744
time: 59.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5283402, upper bound: 27.5254001
time: 63.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7314796, 43.7313957
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4291191, 39.4288712
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8798141, 51.8805847
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2713928, 51.2711372
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.8067169, 55.8073692
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2702980, 35.2699585
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0785217, 44.0782661
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6438255, 49.6442947
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9104843, 69.9112244
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.7936707, 56.7947845
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5663681, 57.5669937
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5218735, 60.5220108
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8660736, 67.8664780
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5840759, 79.5842896
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2898254, 59.2902756
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4138031, 53.4141541
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2447815, 38.2458496

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1539

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 627

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5254001, upper bound: 27.5283402
time: 63.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5133744, upper bound: 27.5404152
time: 70.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7312660, 43.7316093
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4287224, 39.4292755
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8801422, 51.8802528
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2709198, 51.2716103
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.8074188, 55.8066750
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2699623, 35.2702942
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0783539, 44.0784302
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6442833, 49.6438370
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9116745, 69.9100494
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.7954941, 56.7929649
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5675278, 57.5658379
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5219650, 60.5219116
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8666534, 67.8658829
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5845184, 79.5838470
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2902679, 59.2898369
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4141617, 53.4137955
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2452393, 38.2453918

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1539

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 627

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5357339, upper bound: 27.5179921
time: 59.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5237113, upper bound: 27.5300723
time: 58.19 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 120.44 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 120.44
Output dim: 31, lower bound: -27.5300723, upper bound: 27.5237113
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 120.44
Output dim: 31, lower bound: -27.5179921, upper bound: 27.5357339
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 120.44
Output dim: 31, lower bound: -27.5404152, upper bound: 27.5133744
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 120.44
Output dim: 31, lower bound: -27.5283402, upper bound: 27.5254001
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 120.44
Output dim: 31, lower bound: -27.5254001, upper bound: 27.5283402
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 120.44
Output dim: 31, lower bound: -27.5133744, upper bound: 27.5404152
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 120.44
Output dim: 31, lower bound: -27.5357339, upper bound: 27.5179921
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 120.44
Output dim: 31, lower bound: -27.5237113, upper bound: 27.5300723

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7342758, 43.7342148
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4357529, 39.4353218
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8884277, 51.8890572
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2787476, 51.2784004
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.8181763, 55.8188934
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2631302, 35.2624321
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0743256, 44.0739708
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6517334, 49.6526260
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9292297, 69.9304428
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.8212891, 56.8231049
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5841293, 57.5849609
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5243607, 60.5245743
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8760376, 67.8767242
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5902252, 79.5904465
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2957153, 59.2963409
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4206009, 53.4213791
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2585373, 38.2593689

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1539

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5295856, upper bound: 27.5232585
time: 284.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5296257, upper bound: 27.5232043
time: 54.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7345505, 43.7339401
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4358673, 39.4352036
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8891602, 51.8883209
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2790909, 51.2780609
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.8181458, 55.8189201
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2627640, 35.2627983
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0740433, 44.0742455
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6521759, 49.6521797
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9288330, 69.9308319
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.8205795, 56.8238106
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5832748, 57.5858154
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5245132, 60.5244217
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8759460, 67.8768234
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5897675, 79.5909042
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2958984, 59.2961426
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4210129, 53.4209671
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2595139, 38.2583885

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1539

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5174818, upper bound: 27.5353059
time: 59.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5175255, upper bound: 27.5352506
time: 58.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7340698, 43.7344208
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4353485, 39.4357224
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8887634, 51.8887253
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2782745, 51.2788734
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.8188782, 55.8181953
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2627945, 35.2627678
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0741577, 44.0741348
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6521912, 49.6521683
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9303894, 69.9292755
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.8231049, 56.8212852
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5852814, 57.5838051
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5244522, 60.5244751
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8766479, 67.8761292
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5906830, 79.5900040
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2961426, 59.2959023
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4209595, 53.4210205
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2589951, 38.2589111

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1539

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5399319, upper bound: 27.5129245
time: 89.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5399699, upper bound: 27.5128698
time: 1142.89 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 1234.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1234.27
Output dim: 31, lower bound: -27.5295856, upper bound: 27.5232585
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1234.27
Output dim: 31, lower bound: -27.5296257, upper bound: 27.5232043
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1234.27
Output dim: 31, lower bound: -27.5174818, upper bound: 27.5353059
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1234.27
Output dim: 31, lower bound: -27.5175255, upper bound: 27.5352506
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1234.27
Output dim: 31, lower bound: -27.5399319, upper bound: 27.5129245
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1234.27
Output dim: 31, lower bound: -27.5399699, upper bound: 27.5128698
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1234.27
Output dim: 31, lower bound: -27.5283402, upper bound: 27.5254001
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1234.27
Output dim: 31, lower bound: -27.5254001, upper bound: 27.5283402
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1234.27
Output dim: 31, lower bound: -27.5133744, upper bound: 27.5404152
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1234.27
Output dim: 31, lower bound: -27.5357339, upper bound: 27.5179921
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1234.27
Output dim: 31, lower bound: -27.5237113, upper bound: 27.5300723

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 87.82 + 2633.81 = 2721.64 seconds
